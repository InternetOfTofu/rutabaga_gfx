// Copyright 2026 Google
// SPDX-License-Identifier: MIT

use std::io::Error as IoError;
use std::io::ErrorKind;
use std::io::IoSlice;
use std::io::IoSliceMut;
use std::mem::MaybeUninit;
use std::os::fd::AsFd;
use std::path::Path;

use rustix::cmsg_space;
use rustix::fs::fcntl_setfl;
use rustix::fs::OFlags;
use rustix::io::Errno;
use rustix::net::accept;
use rustix::net::bind;
use rustix::net::connect;
use rustix::net::listen;
use rustix::net::recv;
use rustix::net::recvmsg;
use rustix::net::sendmsg;
use rustix::net::socket_with;
use rustix::net::sockopt::socket_type;
use rustix::net::AddressFamily;
use rustix::net::RecvAncillaryBuffer;
use rustix::net::RecvAncillaryMessage;
use rustix::net::RecvFlags;
use rustix::net::SendAncillaryBuffer;
use rustix::net::SendAncillaryMessage;
use rustix::net::SendFlags;
use rustix::net::SocketAddrUnix;
use rustix::net::SocketFlags;
use rustix::net::SocketType;
use rustix::path::Arg;

use crate::util::AsBorrowedDescriptor;
use crate::util::Error;
use crate::util::OwnedDescriptor;
use crate::util::Result as MagmaGpuResult;
use crate::util::TubeType;

const MAX_IDENTIFIERS: usize = 28;
const VPRT_FRAME_MAGIC: u32 = 0x5650_5254;
const VPRT_FRAME_HEADER_LEN: usize = 8;

pub struct Tube {
    socket: OwnedDescriptor,
}

impl TryFrom<OwnedDescriptor> for Tube {
    type Error = Error;

    fn try_from(socket: OwnedDescriptor) -> Result<Self, Self::Error> {
        let ty = socket_type(&socket)?;
        match ty {
            SocketType::SEQPACKET | SocketType::STREAM => Ok(Tube { socket }),
            _ => Err(Error::Unsupported),
        }
    }
}

impl TryFrom<SocketType> for TubeType {
    type Error = IoError;

    fn try_from(ty: SocketType) -> Result<Self, Self::Error> {
        match ty {
            SocketType::SEQPACKET => Ok(TubeType::Packet),
            SocketType::STREAM => Ok(TubeType::Stream),
            ty => {
                log::warn!("Unsupported socket type {ty:?}");
                Err(IoError::from(ErrorKind::Unsupported))
            }
        }
    }
}

impl Tube {
    pub fn new<P: AsRef<Path> + Arg>(path: P, kind: TubeType) -> MagmaGpuResult<Tube> {
        let socket = match kind {
            TubeType::Packet => socket_with(
                AddressFamily::UNIX,
                SocketType::SEQPACKET,
                SocketFlags::empty(),
                None,
            )?,
            TubeType::Stream => socket_with(
                AddressFamily::UNIX,
                SocketType::STREAM,
                SocketFlags::CLOEXEC,
                None,
            )?,
        };

        let unix_addr = SocketAddrUnix::new(path)?;
        connect(&socket, &unix_addr)?;

        Ok(Tube {
            socket: socket.into(),
        })
    }

    pub fn send(
        &self,
        opaque_data: &[u8],
        descriptors: Vec<OwnedDescriptor>,
    ) -> MagmaGpuResult<usize> {
        let mut total_sent = 0usize;
        let mut send_fds = !descriptors.is_empty();

        while total_sent < opaque_data.len() || send_fds {
            let mut space = [MaybeUninit::<u8>::uninit(); cmsg_space!(ScmRights(MAX_IDENTIFIERS))];
            let mut cmsg_buffer = SendAncillaryBuffer::new(&mut space);
            let borrowed_fds: Vec<_> = if send_fds {
                descriptors.iter().map(AsFd::as_fd).collect()
            } else {
                Vec::new()
            };
            if send_fds {
                let cmsg = SendAncillaryMessage::ScmRights(&borrowed_fds);
                cmsg_buffer.push(cmsg);
            }

            let slice = &opaque_data[total_sent..];
            match sendmsg(
                &self.socket,
                &[IoSlice::new(slice)],
                &mut cmsg_buffer,
                SendFlags::NOSIGNAL,
            ) {
                Ok(n) => {
                    send_fds = false;
                    total_sent += n;
                    if opaque_data.is_empty() {
                        break;
                    }
                    if n == 0 {
                        return Err(IoError::new(
                            ErrorKind::WriteZero,
                            "failed to write whole buffer to tube",
                        )
                        .into());
                    }
                }
                Err(Errno::INTR) => continue,
                Err(e) => return Err(e.into()),
            }
        }

        Ok(total_sent)
    }

    pub fn receive(&self, opaque_data: &mut [u8]) -> MagmaGpuResult<(usize, Vec<OwnedDescriptor>)> {
        let is_stream = matches!(socket_type(&self.socket), Ok(SocketType::STREAM));
        let mut expected_frame_len: Option<usize> = None;

        if is_stream && opaque_data.len() >= VPRT_FRAME_HEADER_LEN {
            let mut peek_buf = [0u8; VPRT_FRAME_HEADER_LEN];
            loop {
                match recv(&self.socket, &mut peek_buf, RecvFlags::PEEK) {
                    Ok((n, _)) if n >= VPRT_FRAME_HEADER_LEN => {
                        let magic = u32::from_le_bytes([
                            peek_buf[0],
                            peek_buf[1],
                            peek_buf[2],
                            peek_buf[3],
                        ]);
                        if magic == VPRT_FRAME_MAGIC {
                            let payload_len = u32::from_le_bytes([
                                peek_buf[4],
                                peek_buf[5],
                                peek_buf[6],
                                peek_buf[7],
                            ]) as usize;
                            expected_frame_len = Some(
                                VPRT_FRAME_HEADER_LEN
                                    .saturating_add(payload_len)
                                    .min(opaque_data.len()),
                            );
                        }
                        break;
                    }
                    Ok(_) => break,
                    Err(Errno::INTR) => continue,
                    Err(_) => break,
                }
            }
        }

        let target_len = expected_frame_len.unwrap_or(opaque_data.len());
        let mut received_descriptors: Vec<OwnedDescriptor> = Vec::new();
        let mut total_read = loop {
            let mut iovecs = [IoSliceMut::new(&mut opaque_data[..target_len])];
            let mut space = [MaybeUninit::<u8>::uninit(); cmsg_space!(ScmRights(MAX_IDENTIFIERS))];
            let mut cmsg_buffer = RecvAncillaryBuffer::new(&mut space);
            match recvmsg(
                &self.socket,
                &mut iovecs,
                &mut cmsg_buffer,
                RecvFlags::empty(),
            ) {
                Ok(r) => {
                    for cmsg in cmsg_buffer.drain() {
                        match cmsg {
                            RecvAncillaryMessage::ScmRights(fds) => {
                                received_descriptors.extend(fds.into_iter().map(Into::into));
                            }
                            _ => return Err(Error::Unsupported),
                        }
                    }
                    break r.bytes;
                }
                Err(Errno::INTR) => continue,
                Err(e) => return Err(e.into()),
            }
        };

        if let Some(frame_len) = expected_frame_len {
            while total_read > 0 && total_read < frame_len {
                match recv(
                    &self.socket,
                    &mut opaque_data[total_read..frame_len],
                    RecvFlags::empty(),
                ) {
                    Ok((0, _)) => break,
                    Ok((n, _)) => total_read += n,
                    Err(Errno::INTR) => continue,
                    Err(e) => return Err(e.into()),
                }
            }
        }

        Ok((total_read, received_descriptors))
    }
}

impl AsBorrowedDescriptor for Tube {
    fn as_borrowed_descriptor(&self) -> &OwnedDescriptor {
        &self.socket
    }
}

pub struct Listener {
    socket: OwnedDescriptor,
}

impl Listener {
    /// Creates a new `Listener` bound to the given path.
    pub fn bind<P: AsRef<Path> + Arg>(path: P) -> MagmaGpuResult<Listener> {
        let socket = socket_with(
            AddressFamily::UNIX,
            SocketType::SEQPACKET,
            SocketFlags::empty(),
            None,
        )?;

        let unix_addr = SocketAddrUnix::new(path)?;
        bind(&socket, &unix_addr)?;
        listen(&socket, 128)?;

        fcntl_setfl(&socket, OFlags::NONBLOCK)?;

        Ok(Listener {
            socket: socket.into(),
        })
    }

    pub fn accept(&self) -> MagmaGpuResult<Tube> {
        let accepted_fd = accept(&self.socket)?;
        let descriptor: OwnedDescriptor = accepted_fd.into();
        Ok(Tube { socket: descriptor })
    }
}

impl AsBorrowedDescriptor for Listener {
    fn as_borrowed_descriptor(&self) -> &OwnedDescriptor {
        &self.socket
    }
}
