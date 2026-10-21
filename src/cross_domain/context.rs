// Copyright 2026 Magma-GPU project
// SPDX-License-Identifier: MIT

use std::collections::btree_map::Entry;
use std::collections::BTreeMap as Map;
use std::convert::TryInto;
use std::sync::Arc;
use std::sync::Mutex;
use std::thread;

use log::error;

use magma_gpu::util::create_event_pair;
use magma_gpu::util::create_pipe;
use magma_gpu::util::AsBorrowedDescriptor;
use magma_gpu::util::Error as MagmaGpuError;
use magma_gpu::util::EventSignaler;
use magma_gpu::util::Handle as MagmaGpuHandle;
use magma_gpu::util::MemoryMapping;
use magma_gpu::util::OwnedDescriptor;
use magma_gpu::util::SharedMemory;
use magma_gpu::util::Tube;
use magma_gpu::util::TubeType;
use magma_gpu::util::WaitContext;
use magma_gpu::util::WritePipe;

use zerocopy::FromBytes;
use zerocopy::IntoBytes;

use crate::context_common::ContextResource;
use crate::context_common::ContextResources;
use crate::cross_domain::common::add_item;
use crate::cross_domain::common::CrossDomainItem;
use crate::cross_domain::common::CrossDomainItemState;
use crate::cross_domain::common::CrossDomainJob;
use crate::cross_domain::common::CrossDomainState;
use crate::cross_domain::common::RingWrite;
use crate::cross_domain::common::SentinelManager;
use crate::cross_domain::common::CROSS_DOMAIN_CONTEXT_CHANNEL_ID;
use crate::cross_domain::cross_domain_protocol::CrossDomainAssignSocketUuid;
use crate::cross_domain::cross_domain_protocol::CrossDomainCreateAtomicMemorySentinel;
use crate::cross_domain::cross_domain_protocol::CrossDomainCreateEvent;
use crate::cross_domain::cross_domain_protocol::CrossDomainDestroyAtomicMemorySentinel;
use crate::cross_domain::cross_domain_protocol::CrossDomainGetImageRequirements;
use crate::cross_domain::cross_domain_protocol::CrossDomainHeader;
use crate::cross_domain::cross_domain_protocol::CrossDomainImageRequirements;
use crate::cross_domain::cross_domain_protocol::CrossDomainImportVirtioFsHandle;
use crate::cross_domain::cross_domain_protocol::CrossDomainInit;
use crate::cross_domain::cross_domain_protocol::CrossDomainInitLegacy;
use crate::cross_domain::cross_domain_protocol::CrossDomainInitV1;
use crate::cross_domain::cross_domain_protocol::CrossDomainReadWrite;
use crate::cross_domain::cross_domain_protocol::CrossDomainSendReceive;
use crate::cross_domain::cross_domain_protocol::CrossDomainSignalAtomicMemorySentinel;
use crate::cross_domain::cross_domain_protocol::CROSS_DOMAIN_CHANNEL_RING;
use crate::cross_domain::cross_domain_protocol::CROSS_DOMAIN_CHANNEL_TYPE_INTERNAL_SOCKET;
use crate::cross_domain::cross_domain_protocol::CROSS_DOMAIN_CHANNEL_TYPE_NPU;
use crate::cross_domain::cross_domain_protocol::CROSS_DOMAIN_CMD_ASSIGN_SOCKET_UUID;
use crate::cross_domain::cross_domain_protocol::CROSS_DOMAIN_CMD_CREATE_ATOMIC_MEMORY_SENTINEL;
use crate::cross_domain::cross_domain_protocol::CROSS_DOMAIN_CMD_DESTROY_ATOMIC_MEMORY_SENTINEL;
use crate::cross_domain::cross_domain_protocol::CROSS_DOMAIN_CMD_GET_IMAGE_REQUIREMENTS;
use crate::cross_domain::cross_domain_protocol::CROSS_DOMAIN_CMD_IMPORT_VIRTIOFS_HANDLE;
use crate::cross_domain::cross_domain_protocol::CROSS_DOMAIN_CMD_INIT;
use crate::cross_domain::cross_domain_protocol::CROSS_DOMAIN_CMD_POLL;
use crate::cross_domain::cross_domain_protocol::CROSS_DOMAIN_CMD_READ_CREATE_EVENT;
use crate::cross_domain::cross_domain_protocol::CROSS_DOMAIN_CMD_SEND;
use crate::cross_domain::cross_domain_protocol::CROSS_DOMAIN_CMD_SIGNAL_ATOMIC_MEMORY_SENTINEL;
use crate::cross_domain::cross_domain_protocol::CROSS_DOMAIN_CMD_WRITE;
use crate::cross_domain::cross_domain_protocol::CROSS_DOMAIN_ID_TYPE_READ_PIPE;
use crate::cross_domain::cross_domain_protocol::CROSS_DOMAIN_ID_TYPE_VIRTGPU_BLOB;
use crate::cross_domain::cross_domain_protocol::CROSS_DOMAIN_ID_TYPE_VIRTIO_FS_BLOB;
use crate::cross_domain::cross_domain_protocol::CROSS_DOMAIN_MAX_IDENTIFIERS;
use crate::cross_domain::cross_domain_protocol::CROSS_DOMAIN_QUERY_RING;
use crate::cross_domain::worker::CrossDomainWorker;
use crate::handle::RutabagaHandle;
use crate::resource::RutabagaResource;
use crate::rutabaga_core::RutabagaContext;
use crate::rutabaga_core::VirtioFsLookup;
use crate::rutabaga_utils::Resource3DInfo;
use crate::rutabaga_utils::ResourceCreateBlob;
use crate::rutabaga_utils::RutabagaComponentType;
use crate::rutabaga_utils::RutabagaError;
use crate::rutabaga_utils::RutabagaFence;
use crate::rutabaga_utils::RutabagaFenceHandler;
use crate::rutabaga_utils::RutabagaPath;
use crate::rutabaga_utils::RutabagaResult;
use crate::rutabaga_utils::RUTABAGA_BLOB_MEM_GUEST;
use crate::rutabaga_utils::RUTABAGA_MAP_ACCESS_RW;
use crate::rutabaga_utils::RUTABAGA_MAP_CACHE_CACHED;
use crate::rutabaga_utils::RUTABAGA_PATH_TYPE_NPU;
use crate::DrmFormat;
use crate::ImageAllocationInfo;
use crate::ImageMemoryRequirements;
use crate::RutabagaGralloc;
use crate::RutabagaGrallocFlags;
use crate::RutabagaIovec;

pub struct CrossDomainContext {
    pub paths: Option<Vec<RutabagaPath>>,
    pub gralloc: Arc<Mutex<RutabagaGralloc>>,
    pub state: Option<Arc<CrossDomainState>>,
    pub context_resources: ContextResources,
    pub item_state: CrossDomainItemState,
    pub sentinel_manager: SentinelManager,
    pub fence_handler: RutabagaFenceHandler,
    pub virtiofs_lookup: Option<Arc<dyn VirtioFsLookup>>,
    pub internal_sockets: Arc<Mutex<Map<u128, Tube>>>,
    pub worker_thread: Option<thread::JoinHandle<RutabagaResult<()>>>,
    pub resample_evt: Option<EventSignaler>,
    pub kill_evt: Option<EventSignaler>,
}

impl CrossDomainContext {
    fn get_connection(&mut self, cmd_init: &CrossDomainInit) -> RutabagaResult<Tube> {
        let paths = self
            .paths
            .as_ref()
            .ok_or(RutabagaError::InvalidCrossDomainChannel)?;
        let expected_path_type = match cmd_init.channel_type {
            CROSS_DOMAIN_CHANNEL_TYPE_NPU => RUTABAGA_PATH_TYPE_NPU,
            channel_type => channel_type,
        };
        let path = &paths
            .iter()
            .find(|path| path.path_type == expected_path_type)
            .ok_or(RutabagaError::InvalidCrossDomainChannel)?
            .path;

        let tube = Tube::new(path.clone(), TubeType::Stream)?;
        Ok(tube)
    }

    fn get_internal_socket_connection(&mut self, uuid: u128) -> RutabagaResult<Tube> {
        let mut sockets = self.internal_sockets.lock().unwrap();
        let socket = sockets
            .remove(&uuid)
            .ok_or(RutabagaError::InvalidCrossDomainInternalSocketUuid)?;

        Ok(socket)
    }

    fn initialize(&mut self, cmd_init: &CrossDomainInit) -> RutabagaResult<()> {
        if !self
            .context_resources
            .lock()
            .unwrap()
            .contains_key(&cmd_init.query_ring_id)
        {
            return Err(RutabagaError::InvalidResourceId);
        }

        let query_ring_id = cmd_init.query_ring_id;
        let channel_ring_id = cmd_init.channel_ring_id;
        let context_resources = self.context_resources.clone();
        let sentinel_manager = self.sentinel_manager.clone();

        // Zero means no requested channel.
        if cmd_init.channel_type != 0 {
            if !self
                .context_resources
                .lock()
                .unwrap()
                .contains_key(&cmd_init.channel_ring_id)
            {
                return Err(RutabagaError::InvalidResourceId);
            }

            let connection = if cmd_init.channel_type == CROSS_DOMAIN_CHANNEL_TYPE_INTERNAL_SOCKET {
                self.get_internal_socket_connection(u128::from_le_bytes(
                    cmd_init.internal_socket_uuid,
                ))?
            } else {
                self.get_connection(cmd_init)?
            };

            // This end signals, the worker thread waits.
            let (kill_evt, thread_kill_evt) = create_event_pair()?;
            let (resample_evt, thread_resample_evt) = create_event_pair()?;

            let mut wait_ctx = WaitContext::new()?;
            wait_ctx.add(
                CROSS_DOMAIN_CONTEXT_CHANNEL_ID,
                connection.as_borrowed_descriptor(),
            )?;

            let state = Arc::new(CrossDomainState::new(
                query_ring_id,
                channel_ring_id,
                context_resources,
                sentinel_manager,
                Some(connection),
            ));

            let thread_state = state.clone();
            let thread_items = self.item_state.clone();
            let thread_fence_handler = self.fence_handler.clone();

            let worker_result = thread::Builder::new()
                .name("cross domain".to_string())
                .spawn(move || -> RutabagaResult<()> {
                    CrossDomainWorker::new(
                        wait_ctx,
                        thread_state,
                        thread_items,
                        thread_fence_handler,
                    )
                    .run(thread_kill_evt, thread_resample_evt)
                });

            self.worker_thread = Some(worker_result.map_err(MagmaGpuError::IoError)?);
            self.state = Some(state);
            self.resample_evt = Some(resample_evt);
            self.kill_evt = Some(kill_evt);
        } else {
            self.state = Some(Arc::new(CrossDomainState::new(
                query_ring_id,
                channel_ring_id,
                context_resources,
                sentinel_manager,
                None,
            )));
        }

        Ok(())
    }

    fn get_image_requirements(
        &mut self,
        cmd_get_reqs: &CrossDomainGetImageRequirements,
    ) -> RutabagaResult<()> {
        let info = ImageAllocationInfo {
            width: cmd_get_reqs.width,
            height: cmd_get_reqs.height,
            drm_format: DrmFormat::from(cmd_get_reqs.drm_format),
            flags: RutabagaGrallocFlags::new(cmd_get_reqs.flags),
        };

        let reqs = self
            .gralloc
            .lock()
            .unwrap()
            .get_image_memory_requirements(info)?;

        let mut response = CrossDomainImageRequirements {
            strides: reqs.strides,
            offsets: reqs.offsets,
            modifier: reqs.modifier,
            size: reqs.size,
            blob_id: 0,
            map_info: reqs.map_info,
            memory_idx: -1,
            physical_device_idx: -1,
        };

        if let Some(ref vk_info) = reqs.vulkan_info {
            response.memory_idx = vk_info.memory_idx as i32;
            // We return -1 for now since physical_device_idx is deprecated. If this backend is
            // put back into action, it should be using device_id from the request instead.
            response.physical_device_idx = -1;
        }

        if let Some(state) = &self.state {
            response.blob_id = add_item(&self.item_state, CrossDomainItem::ImageRequirements(reqs));
            state.write_to_ring(RingWrite::Write(response, None), state.query_ring_id)?;
            Ok(())
        } else {
            Err(RutabagaError::InvalidCrossDomainState)
        }
    }

    fn send(
        &mut self,
        cmd_send: &mut CrossDomainSendReceive,
        opaque_data: &[u8],
    ) -> RutabagaResult<()> {
        let mut descriptors: Vec<OwnedDescriptor> = vec![];
        let mut write_pipe_opt: Option<WritePipe> = None;
        let mut read_pipe_id_opt: Option<u32> = None;

        let num_identifiers = cmd_send.num_identifiers as usize;

        if num_identifiers > CROSS_DOMAIN_MAX_IDENTIFIERS {
            return Err(MagmaGpuError::WithContext("max cross domain identifiers exceeded").into());
        }

        let iter = cmd_send
            .identifiers
            .iter_mut()
            .zip(cmd_send.identifier_types.iter_mut())
            .zip(cmd_send.identifier_sizes.iter_mut())
            .map(|((i, it), is)| (i, it, is))
            .take(num_identifiers);

        for (identifier, identifier_type, identifier_size) in iter {
            if *identifier_type == CROSS_DOMAIN_ID_TYPE_VIRTGPU_BLOB {
                let context_resources = self.context_resources.lock().unwrap();

                let context_resource = context_resources
                    .get(identifier)
                    .ok_or(RutabagaError::InvalidResourceId)?;

                if let Some(mesa_handle) = context_resource
                    .handle
                    .as_ref()
                    .and_then(|h| h.as_mesa_handle())
                {
                    descriptors.push(
                        mesa_handle
                            .os_handle
                            .try_clone()
                            .map_err(MagmaGpuError::IoError)?,
                    );
                } else if let Some(ref iovecs) = context_resource.backing_iovecs {
                    let total_iovec_len: usize = iovecs.iter().map(|iov| iov.len).sum();
                    let requested_len = *identifier_size as usize;
                    let copy_len = if requested_len > 0 && requested_len <= total_iovec_len {
                        requested_len
                    } else {
                        total_iovec_len
                    };
                    let shm = SharedMemory::new("cross_domain_guest_blob", copy_len as u64)?;
                    let desc: OwnedDescriptor = shm.into();
                    if copy_len > 0 {
                        let mapping = MemoryMapping::from_offset(&desc, 0, copy_len)?;
                        // SAFETY: `mapping` owns a valid shared memory mapping of `copy_len` bytes,
                        // and each `RutabagaIovec` in `iovecs` was attached to this context.
                        unsafe {
                            let dst = std::slice::from_raw_parts_mut(
                                mapping.as_raw_mapping().ptr as *mut u8,
                                copy_len,
                            );
                            let mut offset = 0usize;
                            for iov in iovecs {
                                if offset >= copy_len {
                                    break;
                                }
                                let chunk_len = (copy_len - offset).min(iov.len);
                                let src =
                                    std::slice::from_raw_parts(iov.base as *const u8, chunk_len);
                                dst[offset..offset + chunk_len].copy_from_slice(src);
                                offset += chunk_len;
                            }
                        }
                    }
                    descriptors.push(desc);
                } else {
                    return Err(MagmaGpuError::InvalidMagmaHandle.into());
                }
            } else if *identifier_type == CROSS_DOMAIN_ID_TYPE_READ_PIPE {
                // In practice, just 1 pipe pair per send is observed.  If we encounter
                // more, this can be changed later.
                if write_pipe_opt.is_some() {
                    return Err(MagmaGpuError::WithContext("expected just one pipe pair").into());
                }

                let (read_pipe, write_pipe) = create_pipe()?;

                descriptors.push(
                    write_pipe
                        .as_borrowed_descriptor()
                        .try_clone()
                        .map_err(MagmaGpuError::IoError)?,
                );
                let read_pipe_id: u32 = add_item(
                    &self.item_state,
                    CrossDomainItem::WaylandReadPipe(read_pipe),
                );

                // For Wayland read pipes, the guest guesses which identifier the host will use to
                // avoid waiting for the host to generate one.  Validate guess here.  This works
                // because of the way Sommelier copy + paste works.  If the Sommelier sequence of
                // events changes, it's always possible to wait for the host
                // response.
                if read_pipe_id != *identifier {
                    return Err(RutabagaError::InvalidCrossDomainItemId);
                }

                // The write pipe needs to be dropped after the send_msg(..) call is complete, so
                // the read pipe can receive subsequent hang-up events.
                write_pipe_opt = Some(write_pipe);
                read_pipe_id_opt = Some(read_pipe_id);
            } else if *identifier_type == CROSS_DOMAIN_ID_TYPE_VIRTIO_FS_BLOB {
                if let Some(CrossDomainItem::RegularFile(file)) =
                    self.item_state.lock().unwrap().table.remove(identifier)
                {
                    descriptors.push(file);
                } else {
                    return Err(RutabagaError::InvalidCrossDomainItemId);
                }
            } else {
                // Don't know how to handle anything else yet.
                return Err(RutabagaError::InvalidCrossDomainItemType);
            }
        }

        if let (Some(state), Some(ref resample_evt)) = (&self.state, &self.resample_evt) {
            state.send_msg(opaque_data, descriptors)?;

            if let Some(read_pipe_id) = read_pipe_id_opt {
                state.add_job(CrossDomainJob::AddReadPipe(read_pipe_id));
                resample_evt.signal()?;
            }
        } else {
            return Err(RutabagaError::InvalidCrossDomainState);
        }

        Ok(())
    }

    fn atomic_memory_sentinel_signal(
        &mut self,
        cmd_atomic_memory_sentinel_signal: &CrossDomainSignalAtomicMemorySentinel,
    ) -> RutabagaResult<()> {
        let manager = self.sentinel_manager.lock().unwrap();
        manager.signal_watcher(cmd_atomic_memory_sentinel_signal.id)
    }

    fn atomic_memory_sentinel_destroy(
        &mut self,
        cmd_atomic_memory_sentinel_destroy: &CrossDomainDestroyAtomicMemorySentinel,
    ) -> RutabagaResult<()> {
        let mut manager = self.sentinel_manager.lock().unwrap();
        manager.destroy_watcher(cmd_atomic_memory_sentinel_destroy.id)
    }

    fn atomic_memory_sentinel_new(
        &mut self,
        cmd_atomic_memory_sentinel_new: &CrossDomainCreateAtomicMemorySentinel,
    ) -> RutabagaResult<()> {
        let id = cmd_atomic_memory_sentinel_new.id;
        let fs_id = cmd_atomic_memory_sentinel_new.fs_id;
        let handle = cmd_atomic_memory_sentinel_new.handle;

        let mut manager = self.sentinel_manager.lock().unwrap();
        let evt = manager.create_watcher(id, fs_id, handle)?;

        let state = self
            .state
            .as_ref()
            .ok_or(RutabagaError::InvalidCrossDomainState)?;
        state.add_job(CrossDomainJob::AddAtomicMemorySentinel(id, evt));

        Ok(())
    }

    fn read_event_new(&mut self, cmd_event_new: &CrossDomainCreateEvent) -> RutabagaResult<()> {
        let items = self.item_state.lock().unwrap();

        if let Some(item) = items.table.get(&cmd_event_new.id) {
            if let CrossDomainItem::Event(..) = item {
                self.state
                    .as_ref()
                    .unwrap()
                    .add_job(CrossDomainJob::AddReadEvent(cmd_event_new.id));
                self.resample_evt.as_ref().unwrap().signal()?;
                Ok(())
            } else {
                Err(RutabagaError::InvalidCrossDomainItemType)
            }
        } else {
            Err(RutabagaError::InvalidCrossDomainItemId)
        }
    }

    fn import_virtiofs_handle(
        &mut self,
        cmd_imp_handle: &CrossDomainImportVirtioFsHandle,
    ) -> RutabagaResult<()> {
        let mut items = self.item_state.lock().unwrap();

        if items.table.contains_key(&cmd_imp_handle.id) {
            return Err(RutabagaError::InvalidCrossDomainItemId);
        }

        let Some(ref lookup) = self.virtiofs_lookup else {
            return Err(RutabagaError::InvalidCrossDomainState);
        };

        let file = lookup.get_exported_descriptor(cmd_imp_handle.fs_id, cmd_imp_handle.handle)?;

        items
            .table
            .insert(cmd_imp_handle.id, CrossDomainItem::RegularFile(file));

        Ok(())
    }

    fn socket_assign_uuid(
        &mut self,
        cmd_exp_socket: &CrossDomainAssignSocketUuid,
    ) -> RutabagaResult<()> {
        let mut items = self.item_state.lock().unwrap();

        let item = items
            .table
            .remove(&cmd_exp_socket.id)
            .ok_or(RutabagaError::InvalidCrossDomainItemId)?;

        if let CrossDomainItem::Socket(fd) = item {
            let mut sockets = self.internal_sockets.lock().unwrap();
            sockets.insert(
                u128::from_le_bytes(cmd_exp_socket.socket_uuid),
                Tube::try_from(fd)?,
            );
            Ok(())
        } else {
            Err(RutabagaError::InvalidCrossDomainItemType)
        }
    }

    fn write(&self, cmd_write: &CrossDomainReadWrite, opaque_data: &[u8]) -> RutabagaResult<()> {
        let len: usize = cmd_write
            .opaque_data_size
            .try_into()
            .map_err(MagmaGpuError::TryFromIntError)?;

        // For Event items (e.g. PipeWire per-period eventfd wakeups), clone
        // the Arc under the lock and perform the actual write lock-free.
        // This avoids serializing every CMD_WRITE behind unrelated
        // item_state operations (such as add_item from process_receive),
        // which can exceed audio period budgets under stream-create churn.
        {
            let items = self.item_state.lock().unwrap();
            let item = items
                .table
                .get(&cmd_write.identifier)
                .ok_or(RutabagaError::InvalidCrossDomainItemId)?;
            if let CrossDomainItem::Event(_, signaler) = item {
                let signaler = Arc::clone(signaler);

                let Ok(bytes) = <[u8; 8]>::try_from(opaque_data) else {
                    return Err(RutabagaError::InvalidCrossDomainWriteLength);
                };

                drop(items);
                signaler.add(u64::from_le_bytes(bytes))?;

                if cmd_write.hang_up != 0 {
                    let mut items = self.item_state.lock().unwrap();
                    items.table.remove(&cmd_write.identifier);
                }

                return Ok(());
            }
        }

        // For non-Event items, use the original remove-and-reinsert pattern.
        let mut items = self.item_state.lock().unwrap();
        let item = items
            .table
            .remove(&cmd_write.identifier)
            .ok_or(RutabagaError::InvalidCrossDomainItemId)?;

        match item {
            CrossDomainItem::WaylandWritePipe(write_pipe) => {
                if len != 0 {
                    write_pipe.write(opaque_data)?;
                }

                if cmd_write.hang_up == 0 {
                    items.table.insert(
                        cmd_write.identifier,
                        CrossDomainItem::WaylandWritePipe(write_pipe),
                    );
                }

                Ok(())
            }
            _ => Err(RutabagaError::InvalidCrossDomainItemType),
        }
    }

    fn process_cmd_send(&mut self, commands: &mut [u8]) -> RutabagaResult<()> {
        let opaque_data_offset = size_of::<CrossDomainSendReceive>();
        let (mut cmd_send, _) = CrossDomainSendReceive::read_from_prefix(commands.as_bytes())
            .map_err(|_| RutabagaError::InvalidCommandBuffer)?;

        let opaque_data = commands
            .get_mut(opaque_data_offset..opaque_data_offset + cmd_send.opaque_data_size as usize)
            .ok_or(RutabagaError::InvalidCommandSize(
                cmd_send.opaque_data_size as usize,
            ))?;

        self.send(&mut cmd_send, opaque_data)?;
        Ok(())
    }

    fn create_blob_from_iovecs(
        resource_id: u32,
        resource_create_blob: ResourceCreateBlob,
        iovecs: Vec<RutabagaIovec>,
    ) -> RutabagaResource {
        RutabagaResource {
            resource_id,
            handle: None,
            blob: true,
            blob_mem: resource_create_blob.blob_mem,
            blob_flags: resource_create_blob.blob_flags,
            map_info: None,
            info_2d: None,
            info_3d: None,
            vulkan_info: None,
            backing_iovecs: Some(iovecs),
            component_mask: 1 << (RutabagaComponentType::CrossDomain as u8),
            size: resource_create_blob.size,
            mapping: None,
        }
    }

    fn create_blob_from_handle(
        &self,
        resource_id: u32,
        resource_create_blob: ResourceCreateBlob,
        handle: MagmaGpuHandle,
    ) -> RutabagaResult<RutabagaResource> {
        let map_access = handle
            .os_handle
            .determine_map_access_mode()
            .map_err(|e| RutabagaError::MagmaGpuError(e.into()))?;
        let map_info = Some(RUTABAGA_MAP_CACHE_CACHED | map_access);

        Ok(RutabagaResource {
            resource_id,
            handle: Some(Arc::new(handle.into())),
            blob: true,
            blob_mem: resource_create_blob.blob_mem,
            blob_flags: resource_create_blob.blob_flags,
            map_info,
            info_2d: None,
            info_3d: None,
            vulkan_info: None,
            backing_iovecs: None,
            component_mask: 1 << (RutabagaComponentType::CrossDomain as u8),
            size: resource_create_blob.size,
            mapping: None,
        })
    }

    fn create_blob_with_requirements(
        &self,
        resource_id: u32,
        resource_create_blob: ResourceCreateBlob,
        handle_opt: Option<RutabagaHandle>,
        reqs: &ImageMemoryRequirements,
    ) -> RutabagaResult<RutabagaResource> {
        if reqs.size != resource_create_blob.size {
            return Err(MagmaGpuError::WithContext("blob size mismatch").into());
        }

        // Strictly speaking, it's against the virtio-gpu spec to allocate memory in the context
        // create blob function, which says "the actual allocation is done via
        // VIRTIO_GPU_CMD_SUBMIT_3D."  However, atomic resource creation is easiest for the
        // cross-domain use case, so whatever.
        let hnd = match handle_opt {
            Some(handle) => handle,
            None => self.gralloc.lock().unwrap().allocate_memory(*reqs)?.into(),
        };

        let info_3d = Resource3DInfo {
            width: reqs.info.width,
            height: reqs.info.height,
            drm_fourcc: reqs.info.drm_format.into(),
            strides: reqs.strides,
            offsets: reqs.offsets,
            modifier: reqs.modifier,
        };

        Ok(RutabagaResource {
            resource_id,
            handle: Some(Arc::new(hnd)),
            blob: true,
            blob_mem: resource_create_blob.blob_mem,
            blob_flags: resource_create_blob.blob_flags,
            map_info: Some(reqs.map_info | RUTABAGA_MAP_ACCESS_RW),
            info_2d: None,
            info_3d: Some(info_3d),
            vulkan_info: reqs.vulkan_info,
            backing_iovecs: None,
            component_mask: 1 << (RutabagaComponentType::CrossDomain as u8),
            size: resource_create_blob.size,
            mapping: None,
        })
    }
}

impl RutabagaContext for CrossDomainContext {
    fn context_create_blob(
        &mut self,
        resource_id: u32,
        resource_create_blob: ResourceCreateBlob,
        iovec_opt: Option<Vec<RutabagaIovec>>,
        handle_opt: Option<RutabagaHandle>,
    ) -> RutabagaResult<RutabagaResource> {
        // This single interface is responsible for some rather different operations,
        // all of which result in the creation of a blob resource.
        match (
            resource_create_blob.blob_id,
            resource_create_blob.blob_mem,
            handle_opt,
        ) {
            // Case 1: Guest memory with iovecs (only available to this process)
            //  This is used for the socket ring buffers.
            (0, RUTABAGA_BLOB_MEM_GUEST, None) => Ok(Self::create_blob_from_iovecs(
                resource_id,
                resource_create_blob,
                iovec_opt.ok_or({
                    MagmaGpuError::WithContext("tried to create guest memory w/o iovecs")
                })?,
            )),
            // Case 2: Guest memory with imported handle (CREATE_GUEST_HANDLE / PRIME import)
            //  It is the caller's responsibility to create such handles in response to
            //  the CREATE_GUEST_HANDLE flag being set by the guest, because the handle
            //  creation is a hypervisor-specific operation (e.g. udmabuf vs gntdev-dmabuf).
            (0, RUTABAGA_BLOB_MEM_GUEST, Some(handle)) => {
                self.create_blob_from_handle(resource_id, resource_create_blob, handle.try_into()?)
            }
            // Cases 3 & 4: Item-backed blobs
            (item_id, _, handle_opt) => {
                let mut items = self.item_state.lock().unwrap();
                let Entry::Occupied(entry) = items.table.entry(item_id as u32) else {
                    return Err(RutabagaError::InvalidCrossDomainItemId);
                };
                match entry.get() {
                    // Case 3: Memory allocated with requirements (by rutabaga or by the caller)
                    //  The requirements blob's lifetime is managed by the guest,
                    //  and one requirements blob can be reused multiple times.
                    CrossDomainItem::ImageRequirements(reqs) => self.create_blob_with_requirements(
                        resource_id,
                        resource_create_blob,
                        handle_opt,
                        reqs,
                    ),
                    // Case 4: Handles received over the socket (e.g. Wayland keymaps)
                    CrossDomainItem::Blob(_) => {
                        let CrossDomainItem::Blob(handle) = entry.remove() else {
                            unreachable!()
                        };
                        self.create_blob_from_handle(resource_id, resource_create_blob, handle)
                    }
                    _ => Err(RutabagaError::InvalidCrossDomainItemType),
                }
            }
        }
    }

    fn submit_cmd(
        &mut self,
        mut commands: &mut [u8],
        _fence_ids: &[u64],
        _shareable_fences: Vec<MagmaGpuHandle>,
    ) -> RutabagaResult<()> {
        while !commands.is_empty() {
            let (hdr, _) = CrossDomainHeader::read_from_prefix(commands)
                .map_err(|_| RutabagaError::InvalidCommandBuffer)?;

            match hdr.cmd {
                CROSS_DOMAIN_CMD_INIT
                    if hdr.cmd_size as usize == size_of::<CrossDomainInitLegacy>() =>
                {
                    let (cmd_init, _) = CrossDomainInitLegacy::read_from_prefix(commands)
                        .map_err(|_e| RutabagaError::InvalidCommandBuffer)?;
                    self.initialize(&cmd_init.upgrade())?;
                }
                CROSS_DOMAIN_CMD_INIT
                    if hdr.cmd_size as usize == size_of::<CrossDomainInitV1>() =>
                {
                    let (cmd_init, _) = CrossDomainInitV1::read_from_prefix(commands)
                        .map_err(|_e| RutabagaError::InvalidCommandBuffer)?;
                    self.initialize(&cmd_init.upgrade())?;
                }
                CROSS_DOMAIN_CMD_INIT if hdr.cmd_size as usize == size_of::<CrossDomainInit>() => {
                    let (cmd_init, _) = CrossDomainInit::read_from_prefix(commands)
                        .map_err(|_e| RutabagaError::InvalidCommandBuffer)?;
                    self.initialize(&cmd_init)?;
                }
                CROSS_DOMAIN_CMD_GET_IMAGE_REQUIREMENTS => {
                    let (cmd_get_reqs, _) =
                        CrossDomainGetImageRequirements::read_from_prefix(commands)
                            .map_err(|_e| RutabagaError::InvalidCommandBuffer)?;

                    self.get_image_requirements(&cmd_get_reqs)?;
                }
                CROSS_DOMAIN_CMD_SEND => {
                    self.process_cmd_send(commands)?;
                }
                CROSS_DOMAIN_CMD_POLL => {
                    // Actual polling is done in the subsequent when creating a fence.
                }
                CROSS_DOMAIN_CMD_WRITE => {
                    let opaque_data_offset = size_of::<CrossDomainReadWrite>();
                    let (cmd_write, _) = CrossDomainReadWrite::read_from_prefix(commands)
                        .map_err(|_e| RutabagaError::InvalidCommandBuffer)?;

                    let opaque_data = commands
                        .get_mut(
                            opaque_data_offset
                                ..opaque_data_offset + cmd_write.opaque_data_size as usize,
                        )
                        .ok_or::<RutabagaError>(RutabagaError::InvalidCommandSize(
                            cmd_write.opaque_data_size as usize,
                        ))?;

                    self.write(&cmd_write, opaque_data)?;
                }
                CROSS_DOMAIN_CMD_CREATE_ATOMIC_MEMORY_SENTINEL => {
                    let (cmd_atomic_memory_sentinel_new, _) =
                        CrossDomainCreateAtomicMemorySentinel::read_from_prefix(commands)
                            .map_err(|_e| RutabagaError::InvalidCommandBuffer)?;
                    self.atomic_memory_sentinel_new(&cmd_atomic_memory_sentinel_new)?;
                }
                CROSS_DOMAIN_CMD_SIGNAL_ATOMIC_MEMORY_SENTINEL => {
                    let (cmd_atomic_memory_sentinel_signal, _) =
                        CrossDomainSignalAtomicMemorySentinel::read_from_prefix(commands)
                            .map_err(|_e| RutabagaError::InvalidCommandBuffer)?;
                    self.atomic_memory_sentinel_signal(&cmd_atomic_memory_sentinel_signal)?;
                }
                CROSS_DOMAIN_CMD_DESTROY_ATOMIC_MEMORY_SENTINEL => {
                    let (cmd_atomic_memory_sentinel_destroy, _) =
                        CrossDomainDestroyAtomicMemorySentinel::read_from_prefix(commands)
                            .map_err(|_e| RutabagaError::InvalidCommandBuffer)?;
                    self.atomic_memory_sentinel_destroy(&cmd_atomic_memory_sentinel_destroy)?;
                }
                CROSS_DOMAIN_CMD_READ_CREATE_EVENT => {
                    let (cmd_new_evt, _) = CrossDomainCreateEvent::read_from_prefix(commands)
                        .map_err(|_| RutabagaError::InvalidCommandBuffer)?;
                    self.read_event_new(&cmd_new_evt)?;
                }
                CROSS_DOMAIN_CMD_ASSIGN_SOCKET_UUID => {
                    let (cmd_exp_socket, _) =
                        CrossDomainAssignSocketUuid::read_from_prefix(commands.as_bytes())
                            .map_err(|_| RutabagaError::InvalidCommandBuffer)?;
                    self.socket_assign_uuid(&cmd_exp_socket)?;
                }
                CROSS_DOMAIN_CMD_IMPORT_VIRTIOFS_HANDLE => {
                    let (cmd_imp_handle, _) =
                        CrossDomainImportVirtioFsHandle::read_from_prefix(commands.as_bytes())
                            .map_err(|_| RutabagaError::InvalidCommandBuffer)?;
                    self.import_virtiofs_handle(&cmd_imp_handle)?;
                }
                _ => return Err(MagmaGpuError::WithContext("invalid cross domain command").into()),
            }

            commands = commands
                .get_mut(hdr.cmd_size as usize..)
                .ok_or(RutabagaError::InvalidCommandSize(hdr.cmd_size as usize))?;
        }

        Ok(())
    }

    fn attach(&mut self, resource: &mut RutabagaResource) {
        self.context_resources.lock().unwrap().insert(
            resource.resource_id,
            ContextResource {
                handle: resource.handle.clone(),
                backing_iovecs: resource.backing_iovecs.clone(),
            },
        );
    }

    fn detach(&mut self, resource: &RutabagaResource) {
        self.context_resources
            .lock()
            .unwrap()
            .remove(&resource.resource_id);
    }

    fn context_create_fence(
        &mut self,
        fence: RutabagaFence,
    ) -> RutabagaResult<Option<MagmaGpuHandle>> {
        match fence.ring_idx as u32 {
            CROSS_DOMAIN_QUERY_RING => self.fence_handler.call(fence),
            CROSS_DOMAIN_CHANNEL_RING => {
                if let Some(state) = &self.state {
                    state.add_job(CrossDomainJob::HandleFence(fence));
                }
            }
            _ => return Err(MagmaGpuError::WithContext("unexpected ring type").into()),
        }

        Ok(None)
    }

    fn component_type(&self) -> RutabagaComponentType {
        RutabagaComponentType::CrossDomain
    }
}

impl Drop for CrossDomainContext {
    fn drop(&mut self) {
        if let Some(state) = &self.state {
            state.add_job(CrossDomainJob::Finish);
        }

        if let Some(kill_evt) = self.kill_evt.take() {
            // Log the error, but still try to join the worker thread
            match kill_evt.signal() {
                Ok(_) => (),
                Err(e) => {
                    error!("failed to write cross domain kill event: {e}");
                }
            }

            if let Some(worker_thread) = self.worker_thread.take() {
                let _ = worker_thread.join();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::ffi::c_void;
    use std::mem::size_of;
    use std::os::unix::io::OwnedFd;
    use std::os::unix::net::UnixListener;
    use std::path::PathBuf;
    use std::sync::Condvar;
    use std::time::Duration;

    use super::*;
    use crate::cross_domain::component::CrossDomain;
    use crate::cross_domain::cross_domain_protocol::CrossDomainCapabilities;
    use crate::cross_domain::cross_domain_protocol::CROSS_DOMAIN_CHANNEL_TYPE_PIPEWIRE;
    use crate::cross_domain::cross_domain_protocol::CROSS_DOMAIN_CMD_RECEIVE;
    use crate::rutabaga_utils::RutabagaHandler;
    use crate::rutabaga_utils::RutabagaIovec;
    use crate::rutabaga_utils::RUTABAGA_BLOB_FLAG_USE_MAPPABLE;
    use crate::rutabaga_utils::RUTABAGA_BLOB_MEM_HOST3D;
    use crate::rutabaga_utils::RUTABAGA_FLAG_INFO_RING_IDX;
    use crate::rutabaga_utils::RUTABAGA_PATH_TYPE_PIPEWIRE;

    #[test]
    fn test_cross_domain_npu_channel_connects() {
        assert_eq!(CROSS_DOMAIN_CHANNEL_TYPE_NPU, 0x0014);
        assert_eq!(RUTABAGA_PATH_TYPE_NPU, 0x0014);
        assert_ne!(
            CROSS_DOMAIN_CHANNEL_TYPE_NPU,
            CROSS_DOMAIN_CHANNEL_TYPE_PIPEWIRE
        );
        assert_ne!(RUTABAGA_PATH_TYPE_NPU, RUTABAGA_PATH_TYPE_PIPEWIRE);

        let sock_path = PathBuf::from(format!("/tmp/rutabaga_npu_test_{}.sock", std::process::id()));
        let _ = std::fs::remove_file(&sock_path);
        let listener = UnixListener::bind(&sock_path).expect("failed to bind test unix socket");

        let paths = vec![RutabagaPath {
            path: sock_path.clone(),
            path_type: RUTABAGA_PATH_TYPE_NPU,
        }];
        let fence_handler = RutabagaHandler::new(|_| {});
        let component = CrossDomain::init(Some(paths), fence_handler.clone(), None)
            .expect("failed to init CrossDomain");
        let caps_bytes = component.get_capset(0, 0);
        let (caps, _) =
            CrossDomainCapabilities::read_from_prefix(&caps_bytes).expect("valid capset");
        assert_eq!(caps.supported_channels, 1 << RUTABAGA_PATH_TYPE_NPU);
        let mut ctx = component
            .create_context(1, 0, None, fence_handler)
            .expect("failed to create CrossDomainContext");

        let mut query_ring = RutabagaResource {
            resource_id: 1,
            handle: None,
            blob: true,
            blob_mem: RUTABAGA_BLOB_MEM_GUEST,
            blob_flags: 0,
            map_info: None,
            info_2d: None,
            info_3d: None,
            vulkan_info: None,
            backing_iovecs: Some(Vec::new()),
            component_mask: 0,
            size: 4096,
            mapping: None,
        };
        let mut channel_ring = RutabagaResource {
            resource_id: 2,
            handle: None,
            blob: true,
            blob_mem: RUTABAGA_BLOB_MEM_GUEST,
            blob_flags: 0,
            map_info: None,
            info_2d: None,
            info_3d: None,
            vulkan_info: None,
            backing_iovecs: Some(Vec::new()),
            component_mask: 0,
            size: 4096,
            mapping: None,
        };
        ctx.attach(&mut query_ring);
        ctx.attach(&mut channel_ring);

        let cmd_init = CrossDomainInit {
            hdr: CrossDomainHeader {
                cmd: CROSS_DOMAIN_CMD_INIT,
                ring_idx: 0,
                cmd_size: size_of::<CrossDomainInit>() as u16,
                pad: 0,
            },
            query_ring_id: 1,
            channel_ring_id: 2,
            channel_type: CROSS_DOMAIN_CHANNEL_TYPE_NPU,
            internal_socket_uuid: [0; 16],
        };
        let mut cmd_bytes = cmd_init.as_bytes().to_vec();
        ctx.submit_cmd(&mut cmd_bytes, &[], Vec::new())
            .expect("CROSS_DOMAIN_CMD_INIT for NPU channel should succeed");

        let (_stream, _) = listener
            .accept()
            .expect("NPU socket listener should accept connection");
        drop(ctx);
        let _ = std::fs::remove_file(&sock_path);
    }

    #[test]
    fn test_cross_domain_npu_send_and_receive_with_fds() {
        let sock_path = PathBuf::from(format!(
            "/tmp/rutabaga_npu_roundtrip_{}.sock",
            std::process::id()
        ));
        let _ = std::fs::remove_file(&sock_path);
        let listener = UnixListener::bind(&sock_path).expect("failed to bind test unix socket");

        let signaled_fences: Arc<(Mutex<Vec<u64>>, Condvar)> =
            Arc::new((Mutex::new(Vec::new()), Condvar::new()));
        let signaled_clone = Arc::clone(&signaled_fences);
        let fence_handler = RutabagaHandler::new(move |fence: RutabagaFence| {
            let (lock, cvar) = &*signaled_clone;
            lock.lock().unwrap().push(fence.fence_id);
            cvar.notify_all();
        });

        let paths = vec![RutabagaPath {
            path: sock_path.clone(),
            path_type: RUTABAGA_PATH_TYPE_NPU,
        }];
        let component = CrossDomain::init(Some(paths), fence_handler.clone(), None)
            .expect("failed to init CrossDomain");
        let mut ctx = component
            .create_context(1, 0, None, fence_handler)
            .expect("failed to create CrossDomainContext");

        let mut query_ring_buf = vec![0u8; 4096];
        let mut channel_ring_buf = vec![0u8; 4096];

        let mut query_ring = RutabagaResource {
            resource_id: 1,
            handle: None,
            blob: true,
            blob_mem: RUTABAGA_BLOB_MEM_GUEST,
            blob_flags: 0,
            map_info: None,
            info_2d: None,
            info_3d: None,
            vulkan_info: None,
            backing_iovecs: Some(vec![RutabagaIovec {
                base: query_ring_buf.as_mut_ptr() as *mut c_void,
                len: query_ring_buf.len(),
            }]),
            component_mask: 0,
            size: query_ring_buf.len() as u64,
            mapping: None,
        };
        let mut channel_ring = RutabagaResource {
            resource_id: 2,
            handle: None,
            blob: true,
            blob_mem: RUTABAGA_BLOB_MEM_GUEST,
            blob_flags: 0,
            map_info: None,
            info_2d: None,
            info_3d: None,
            vulkan_info: None,
            backing_iovecs: Some(vec![RutabagaIovec {
                base: channel_ring_buf.as_mut_ptr() as *mut c_void,
                len: channel_ring_buf.len(),
            }]),
            component_mask: 0,
            size: channel_ring_buf.len() as u64,
            mapping: None,
        };
        ctx.attach(&mut query_ring);
        ctx.attach(&mut channel_ring);

        // 1. Initialize CROSS_DOMAIN_CHANNEL_TYPE_NPU
        let cmd_init = CrossDomainInit {
            hdr: CrossDomainHeader {
                cmd: CROSS_DOMAIN_CMD_INIT,
                ring_idx: 0,
                cmd_size: size_of::<CrossDomainInit>() as u16,
                pad: 0,
            },
            query_ring_id: 1,
            channel_ring_id: 2,
            channel_type: CROSS_DOMAIN_CHANNEL_TYPE_NPU,
            internal_socket_uuid: [0; 16],
        };
        let mut init_bytes = cmd_init.as_bytes().to_vec();
        ctx.submit_cmd(&mut init_bytes, &[], Vec::new())
            .expect("CROSS_DOMAIN_CMD_INIT should succeed");

        let (peer_stream, _) = listener.accept().expect("peer should accept connection");
        let peer_fd: OwnedFd = peer_stream.into();
        let peer_tube =
            Tube::try_from(OwnedDescriptor::from(peer_fd)).expect("Tube::try_from should succeed");

        // 2(a). Submit CROSS_DOMAIN_CMD_SEND with 0 identifiers
        let payload_no_fds = b"hello-npu-no-fds";
        let cmd_send_0 = CrossDomainSendReceive {
            hdr: CrossDomainHeader {
                cmd: CROSS_DOMAIN_CMD_SEND,
                ring_idx: 0,
                cmd_size: (size_of::<CrossDomainSendReceive>() + payload_no_fds.len()) as u16,
                pad: 0,
            },
            num_identifiers: 0,
            opaque_data_size: payload_no_fds.len() as u32,
            identifiers: [0; CROSS_DOMAIN_MAX_IDENTIFIERS],
            identifier_types: [0; CROSS_DOMAIN_MAX_IDENTIFIERS],
            identifier_sizes: [0; CROSS_DOMAIN_MAX_IDENTIFIERS],
        };
        let mut send_0_bytes = cmd_send_0.as_bytes().to_vec();
        send_0_bytes.extend_from_slice(payload_no_fds);
        ctx.submit_cmd(&mut send_0_bytes, &[], Vec::new())
            .expect("CROSS_DOMAIN_CMD_SEND with 0 identifiers should succeed");

        let mut recv_buf = vec![0u8; 256];
        let (n0, fds0) = peer_tube
            .receive(&mut recv_buf)
            .expect("peer_tube.receive (0 fds) should succeed");
        assert_eq!(&recv_buf[..n0], payload_no_fds);
        assert!(fds0.is_empty());

        // 2(b). Attach a guest-backed blob resource and submit CROSS_DOMAIN_CMD_SEND with 1 blob
        let mut guest_blob_data = b"guest-blob-payload-1234".to_vec();
        let mut guest_blob_res = RutabagaResource {
            resource_id: 3,
            handle: None,
            blob: true,
            blob_mem: RUTABAGA_BLOB_MEM_GUEST,
            blob_flags: RUTABAGA_BLOB_FLAG_USE_MAPPABLE,
            map_info: None,
            info_2d: None,
            info_3d: None,
            vulkan_info: None,
            backing_iovecs: Some(vec![RutabagaIovec {
                base: guest_blob_data.as_mut_ptr() as *mut c_void,
                len: guest_blob_data.len(),
            }]),
            component_mask: 0,
            size: guest_blob_data.len() as u64,
            mapping: None,
        };
        ctx.attach(&mut guest_blob_res);

        let payload_with_fd = b"invoke-with-blob";
        let mut cmd_send_1 = CrossDomainSendReceive {
            hdr: CrossDomainHeader {
                cmd: CROSS_DOMAIN_CMD_SEND,
                ring_idx: 0,
                cmd_size: (size_of::<CrossDomainSendReceive>() + payload_with_fd.len()) as u16,
                pad: 0,
            },
            num_identifiers: 1,
            opaque_data_size: payload_with_fd.len() as u32,
            identifiers: [0; CROSS_DOMAIN_MAX_IDENTIFIERS],
            identifier_types: [0; CROSS_DOMAIN_MAX_IDENTIFIERS],
            identifier_sizes: [0; CROSS_DOMAIN_MAX_IDENTIFIERS],
        };
        cmd_send_1.identifiers[0] = 3;
        cmd_send_1.identifier_types[0] = CROSS_DOMAIN_ID_TYPE_VIRTGPU_BLOB;
        cmd_send_1.identifier_sizes[0] = guest_blob_data.len() as u32;
        let mut send_1_bytes = cmd_send_1.as_bytes().to_vec();
        send_1_bytes.extend_from_slice(payload_with_fd);
        ctx.submit_cmd(&mut send_1_bytes, &[], Vec::new())
            .expect("CROSS_DOMAIN_CMD_SEND with guest blob should succeed");

        let (n1, fds1) = peer_tube
            .receive(&mut recv_buf)
            .expect("peer_tube.receive (1 fd) should succeed");
        assert_eq!(&recv_buf[..n1], payload_with_fd);
        assert_eq!(fds1.len(), 1);
        let mapped_guest_blob =
            MemoryMapping::from_offset(&fds1[0], 0, guest_blob_data.len()).unwrap();
        let received_blob_slice = unsafe {
            std::slice::from_raw_parts(
                mapped_guest_blob.as_raw_mapping().ptr as *const u8,
                guest_blob_data.len(),
            )
        };
        assert_eq!(received_blob_slice, guest_blob_data.as_slice());

        // 3. Send a response + host memfd FD back from the Unix socket peer, poll, and import blob
        let host_reply_payload = b"host-reply-ok";
        let host_out_bytes = b"host-output-tensor-data";
        let host_shm =
            SharedMemory::new("host_out_blob", host_out_bytes.len() as u64).expect("host shm");
        let host_shm_desc: OwnedDescriptor = host_shm.into();
        {
            let host_map =
                MemoryMapping::from_offset(&host_shm_desc, 0, host_out_bytes.len()).unwrap();
            let dst = unsafe {
                std::slice::from_raw_parts_mut(
                    host_map.as_raw_mapping().ptr as *mut u8,
                    host_out_bytes.len(),
                )
            };
            dst.copy_from_slice(host_out_bytes);
        }
        peer_tube
            .send(host_reply_payload, vec![host_shm_desc])
            .expect("peer_tube.send response + fd should succeed");

        let cmd_poll = CrossDomainHeader {
            cmd: CROSS_DOMAIN_CMD_POLL,
            ring_idx: CROSS_DOMAIN_CHANNEL_RING as u8,
            cmd_size: size_of::<CrossDomainHeader>() as u16,
            pad: 0,
        };
        let mut poll_bytes = cmd_poll.as_bytes().to_vec();
        ctx.submit_cmd(&mut poll_bytes, &[], Vec::new())
            .expect("CROSS_DOMAIN_CMD_POLL should succeed");
        ctx.context_create_fence(RutabagaFence {
            flags: RUTABAGA_FLAG_INFO_RING_IDX,
            fence_id: 42,
            ctx_id: 1,
            ring_idx: CROSS_DOMAIN_CHANNEL_RING as u8,
        })
        .expect("context_create_fence should succeed");

        {
            let (lock, cvar) = &*signaled_fences;
            let mut fences = lock.lock().unwrap();
            while !fences.contains(&42) {
                let (next, timeout) = cvar
                    .wait_timeout(fences, Duration::from_secs(5))
                    .expect("wait_timeout failed");
                fences = next;
                assert!(!timeout.timed_out(), "timed out waiting for channel fence 42");
            }
        }

        let (cmd_recv, rest) =
            CrossDomainSendReceive::read_from_prefix(channel_ring_buf.as_slice())
                .expect("should read CrossDomainSendReceive from channel_ring");
        assert_eq!(cmd_recv.hdr.cmd, CROSS_DOMAIN_CMD_RECEIVE);
        assert_eq!(cmd_recv.num_identifiers, 1);
        assert_eq!(
            cmd_recv.identifier_types[0],
            CROSS_DOMAIN_ID_TYPE_VIRTGPU_BLOB
        );
        assert_eq!(cmd_recv.identifier_sizes[0], host_out_bytes.len() as u32);
        assert_eq!(
            &rest[..cmd_recv.opaque_data_size as usize],
            host_reply_payload
        );

        let imported_res = ctx
            .context_create_blob(
                99,
                ResourceCreateBlob {
                    blob_mem: RUTABAGA_BLOB_MEM_HOST3D,
                    blob_flags: RUTABAGA_BLOB_FLAG_USE_MAPPABLE,
                    blob_id: cmd_recv.identifiers[0] as u64,
                    size: host_out_bytes.len() as u64,
                },
                None,
                None,
            )
            .expect("context_create_blob should import returned host blob");
        let imported_handle = imported_res
            .handle
            .as_ref()
            .and_then(|h| h.as_mesa_handle())
            .expect("imported resource should have mesa handle");
        let imported_map =
            MemoryMapping::from_offset(&imported_handle.os_handle, 0, host_out_bytes.len())
                .expect("should map imported host blob");
        let imported_slice = unsafe {
            std::slice::from_raw_parts(
                imported_map.as_raw_mapping().ptr as *const u8,
                host_out_bytes.len(),
            )
        };
        assert_eq!(imported_slice, host_out_bytes);

        drop(ctx);
        let _ = std::fs::remove_file(&sock_path);
    }

    #[test]
    fn test_tube_stream_vprt_framing_no_coalescing() {
        let sock_path = PathBuf::from(format!(
            "/tmp/rutabaga_tube_vprt_{}.sock",
            std::process::id()
        ));
        let _ = std::fs::remove_file(&sock_path);
        let listener = UnixListener::bind(&sock_path).expect("failed to bind unix socket");

        let client_tube = Tube::new(&sock_path, TubeType::Stream).expect("Tube::new should work");
        let (server_stream, _) = listener.accept().expect("accept should work");
        let server_fd: OwnedFd = server_stream.into();
        let server_tube =
            Tube::try_from(OwnedDescriptor::from(server_fd)).expect("Tube::try_from should work");

        let build_vprt_frame = |payload: &[u8]| -> Vec<u8> {
            let mut frame = Vec::with_capacity(8 + payload.len());
            frame.extend_from_slice(&0x5650_5254u32.to_le_bytes());
            frame.extend_from_slice(&(payload.len() as u32).to_le_bytes());
            frame.extend_from_slice(payload);
            frame
        };

        let frame1 = build_vprt_frame(b"first-framed-response");
        let frame2 = build_vprt_frame(b"second-framed-response-longer");

        let shm1: OwnedDescriptor = SharedMemory::new("shm1", 16).unwrap().into();
        server_tube.send(&frame1, vec![shm1]).unwrap();
        server_tube.send(&frame2, vec![]).unwrap();

        let mut buf = vec![0u8; 4096];
        let (n1, fds1) = client_tube.receive(&mut buf).unwrap();
        assert_eq!(&buf[..n1], frame1.as_slice());
        assert_eq!(fds1.len(), 1);

        let (n2, fds2) = client_tube.receive(&mut buf).unwrap();
        assert_eq!(&buf[..n2], frame2.as_slice());
        assert!(fds2.is_empty());

        let _ = std::fs::remove_file(&sock_path);
    }
}
