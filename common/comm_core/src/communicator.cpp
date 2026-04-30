#include <torch/extension.h>
#include "communicator.h"

/* 无用
void bind_local_gpu() {
    int world_rank, local_rank;
    MPI_Comm local_comm;

    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    // 把同一节点的进程分到一个 communicator
    MPI_Comm_split_type(
        MPI_COMM_WORLD,
        MPI_COMM_TYPE_SHARED,
        0,
        MPI_INFO_NULL,
        &local_comm
    );

    MPI_Comm_rank(local_comm, &local_rank);

    int num_gpus = 0;
    cudaGetDeviceCount(&num_gpus);

    // 防御式检查
    if (num_gpus == 0) {
        fprintf(stderr, "No CUDA devices found\n");
        MPI_Abort(MPI_COMM_WORLD, -1);
    }

    if (local_rank >= num_gpus) {
        fprintf(stderr,
            "local_rank %d >= num_gpus %d (oversubscription?)\n",
            local_rank, num_gpus);
        MPI_Abort(MPI_COMM_WORLD, -1);
    }

    cudaSetDevice(local_rank);
}
*/

void g_init() {
    MPICHECK(MPI_Init(NULL, NULL));
    // bind_local_gpu();   // 新增，无用
}

int g_rank() {
    int r;
    MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &r));
    return r;
}

int g_size() {
    int s;
    MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &s));
    return s;
}

void g_barriar() {
    MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
}

Communicator::Communicator(int nstreams=1) {
    m_current_comm = 0;
    m_num_comms = nstreams;
    m_rank = g_rank();
    m_size = g_size();
    m_destroyed = true;
    _init();
}

Communicator::~Communicator() {
    destroy();
    int finalized;
    MPICHECK(MPI_Finalized(&finalized));
    if (!finalized) {
        MPICHECK(MPI_Finalize());
    }
}

void Communicator::_init() {
    if (!m_destroyed) {
        return;
    }
    int nstreams = m_num_comms;
    m_nccl_ids = new ncclUniqueId[nstreams];
    m_streams = new cudaStream_t[nstreams];
    m_events = new cudaEvent_t[nstreams];
    m_nccl_comms = new ncclComm_t[nstreams];
    m_sync_flags = new bool[nstreams];
    CUDACHECK(cudaEventCreate(&m_op_base_event));
    m_op_base_event_recorded = false;
    for (int i = 0; i < m_num_comms; i++) {
	    if (m_rank == 0) ncclGetUniqueId(&m_nccl_ids[i]);
	    MPICHECK(MPI_Bcast((void *)&m_nccl_ids[i], sizeof(m_nccl_ids[i]), MPI_BYTE, 0, MPI_COMM_WORLD));
    }
    for (int i = 0; i < m_num_comms; i++) {
        //CUDACHECK(cudaStreamCreate(&m_streams[i]));
        at::cuda::CUDAStream myStream = at::cuda::getStreamFromPool();
        m_streams[i] = myStream.stream();
        m_torchstreams.push_back(myStream);
        CUDACHECK(cudaEventCreate(&m_events[i]));
        m_sync_flags[i] = true;
        
        int dev;
        cudaGetDevice(&dev);
        // Dubug：可以看rank和GPU的关系
        // printf("[Rank %d] sees GPU %d\n", m_rank, dev);
        // printf("[Rank %d] , world=%d\n", m_rank, m_size);
        NCCLCHECK(ncclCommInitRank(&m_nccl_comms[i], m_size, m_nccl_ids[i], m_rank));
    }
    m_destroyed = false;
}

void Communicator::destroy() {
    for (auto &event : m_op_start_events) {
        if (event != nullptr) {
            CUDACHECK(cudaEventDestroy(event));
            event = nullptr;
        }
    }
    m_op_start_events.clear();
    for (auto &event : m_op_events) {
        if (event != nullptr) {
            CUDACHECK(cudaEventDestroy(event));
            event = nullptr;
        }
    }
    m_op_events.clear();
    CUDACHECK(cudaEventDestroy(m_op_base_event));
    for (int i = 0; i < m_num_comms; i++) {
        NCCLCHECK(ncclCommDestroy(m_nccl_comms[i]));
        //CUDACHECK(cudaStreamDestroy(m_streams[i]));
        CUDACHECK(cudaEventDestroy(m_events[i]));
    }
    m_destroyed = true;
    delete m_streams;
    delete m_events;
    delete m_nccl_comms;
    delete m_sync_flags;
}

void Communicator::reload() {
    _init();
}

void Communicator::_extendComms(int n_comms) {
    if (m_num_comms >= n_comms) return;
    for (int i = 0; i < n_comms-m_num_comms; i++) {
	    if (m_rank == 0) ncclGetUniqueId(&m_nccl_ids[i+m_num_comms]);
	    MPICHECK(MPI_Bcast((void *)&m_nccl_ids[i+m_num_comms], sizeof(m_nccl_ids[i+m_num_comms]), MPI_BYTE, 0, MPI_COMM_WORLD));

	    CUDACHECK(cudaStreamCreate(&m_streams[i+m_num_comms]));
	    NCCLCHECK(ncclCommInitRank(&m_nccl_comms[i+m_num_comms], m_size, m_nccl_ids[i+m_num_comms], m_rank));
    }
    m_num_comms = n_comms;
}

void Communicator::barrier() {

    MPICHECK(MPI_Barrier(MPI_COMM_WORLD));

}

void Communicator::synchronize() {
    //CUDACHECK(cudaStreamSynchronize(m_stream));
    for (int i = 0; i < m_num_comms; i++) {
        CUDACHECK(cudaStreamSynchronize(m_streams[i]));
        //CUDACHECK(cudaStreamWaitEvent(m_streams[i], m_events[i], 0));
        m_sync_flags[i] = true;
    }
    clearEvents();
}
void Communicator::syncStream(int handler) {
    if (handler < m_num_comms && !m_sync_flags[handler]) {
        CUDACHECK(cudaStreamSynchronize(m_streams[handler]));
        m_sync_flags[handler] = true;
    }
}
void Communicator::syncEvent(int handler) {
    if (handler >= 0 && handler < static_cast<int>(m_op_events.size()) && m_op_events[handler] != nullptr) {
        CUDACHECK(cudaEventSynchronize(m_op_events[handler]));
        if (handler < static_cast<int>(m_op_start_events.size()) && m_op_start_events[handler] != nullptr) {
            CUDACHECK(cudaEventDestroy(m_op_start_events[handler]));
            m_op_start_events[handler] = nullptr;
        }
        CUDACHECK(cudaEventDestroy(m_op_events[handler]));
        m_op_events[handler] = nullptr;
    }
}

float Communicator::syncEventElapsedFromBase(int handler) {
    std::vector<float> elapsed_range_ms = syncEventElapsedRangeFromBase(handler);
    if (elapsed_range_ms.size() >= 2) {
        return elapsed_range_ms[1];
    }
    return -1.0f;
}

std::vector<float> Communicator::syncEventElapsedRangeFromBase(int handler) {
    float start_elapsed_ms = -1.0f;
    float complete_elapsed_ms = -1.0f;
    if (handler >= 0 && handler < static_cast<int>(m_op_events.size()) && m_op_events[handler] != nullptr) {
        CUDACHECK(cudaEventSynchronize(m_op_events[handler]));
        if (m_op_base_event_recorded) {
            if (handler < static_cast<int>(m_op_start_events.size()) && m_op_start_events[handler] != nullptr) {
                CUDACHECK(cudaEventElapsedTime(&start_elapsed_ms, m_op_base_event, m_op_start_events[handler]));
            }
            CUDACHECK(cudaEventElapsedTime(&complete_elapsed_ms, m_op_base_event, m_op_events[handler]));
        }
        if (handler < static_cast<int>(m_op_start_events.size()) && m_op_start_events[handler] != nullptr) {
            CUDACHECK(cudaEventDestroy(m_op_start_events[handler]));
            m_op_start_events[handler] = nullptr;
        }
        CUDACHECK(cudaEventDestroy(m_op_events[handler]));
        m_op_events[handler] = nullptr;
    }
    return {start_elapsed_ms, complete_elapsed_ms};
}

void Communicator::clearEvents() {
    for (auto &event : m_op_start_events) {
        if (event != nullptr) {
            CUDACHECK(cudaEventDestroy(event));
            event = nullptr;
        }
    }
    m_op_start_events.clear();
    for (auto &event : m_op_events) {
        if (event != nullptr) {
            CUDACHECK(cudaEventDestroy(event));
            event = nullptr;
        }
    }
    m_op_events.clear();
    m_op_base_event_recorded = false;
    for (int i = 0; i < m_num_comms; i++) {
        m_sync_flags[i] = true;
    }
}

int Communicator::getNumOfFreeStreams() {
    int num_free_streams = 0;
    for (int i = 0; i < m_num_comms; i++) {
        if (m_sync_flags[i]) num_free_streams++;
        else {
            cudaError_t status = cudaStreamQuery(m_streams[i]);
            if (status == cudaSuccess) num_free_streams++;
        }
    }
    return num_free_streams;
}

int Communicator::reduce(torch::Tensor tensor, int root) {
    int current_comm = m_current_comm;
    NCCLCHECK(ncclReduce(tensor.data_ptr<float>(), tensor.data_ptr<float>(), tensor.numel(), ncclFloat, ncclSum, root, m_nccl_comms[current_comm], m_streams[current_comm]));
    //CUDACHECK(cudaEventRecord(m_events[current_comm], m_streams[current_comm]));
    m_current_comm++;
    m_current_comm %= m_num_comms;
    m_sync_flags[current_comm] = false;
    return current_comm;
}

int Communicator::bcast(torch::Tensor tensor, int root) {
    ncclDataType_t nccl_type;
    int current_comm = m_current_comm;
    if (torch::kFloat32 == tensor.dtype()) {
        nccl_type = ncclFloat;
        NCCLCHECK(ncclBroadcast(tensor.data_ptr<float>(), tensor.data_ptr<float>(), tensor.numel(), nccl_type, root, m_nccl_comms[m_current_comm], m_streams[m_current_comm]));
    } else if (torch::kInt64 == tensor.dtype()) {
        nccl_type = ncclInt64;
        NCCLCHECK(ncclBroadcast(tensor.data_ptr<long>(), tensor.data_ptr<long>(), tensor.numel(), nccl_type, root, m_nccl_comms[m_current_comm], m_streams[m_current_comm]));
    }
    //CUDACHECK(cudaEventRecord(m_events[m_current_comm], m_streams[m_current_comm]));
    m_current_comm++;
    m_current_comm %= m_num_comms;
    m_sync_flags[current_comm] = false;
    return current_comm;
}

int Communicator::reduceScatter(torch::Tensor send_tensor, torch::Tensor recv_tensor) {
    int current_comm = m_current_comm;
    if (!m_op_base_event_recorded) {
        CUDACHECK(cudaEventRecord(m_op_base_event, m_streams[current_comm]));
        m_op_base_event_recorded = true;
    }
    cudaEvent_t start_event;
    CUDACHECK(cudaEventCreate(&start_event));
    CUDACHECK(cudaEventRecord(start_event, m_streams[current_comm]));
    if (torch::kFloat32 == send_tensor.dtype()) {
        NCCLCHECK(ncclReduceScatter(send_tensor.data_ptr<float>(), recv_tensor.data_ptr<float>(), recv_tensor.numel(), ncclFloat, ncclSum, m_nccl_comms[current_comm], m_streams[current_comm]));
    } else if (torch::kInt64 == send_tensor.dtype()) {
        NCCLCHECK(ncclReduceScatter(send_tensor.data_ptr<long>(), recv_tensor.data_ptr<long>(), recv_tensor.numel(), ncclInt64, ncclSum, m_nccl_comms[current_comm], m_streams[current_comm]));
    }
    cudaEvent_t event;
    CUDACHECK(cudaEventCreate(&event));
    CUDACHECK(cudaEventRecord(event, m_streams[current_comm]));
    m_op_start_events.push_back(start_event);
    m_op_events.push_back(event);
    int handler = static_cast<int>(m_op_events.size()) - 1;
    m_current_comm++;
    m_current_comm %= m_num_comms;
    m_sync_flags[current_comm] = false;
    return handler;
}

int Communicator::allGather(torch::Tensor send_tensor, torch::Tensor recv_tensor) {
    int current_comm = m_current_comm;
    if (!m_op_base_event_recorded) {
        CUDACHECK(cudaEventRecord(m_op_base_event, m_streams[current_comm]));
        m_op_base_event_recorded = true;
    }
    cudaEvent_t start_event;
    CUDACHECK(cudaEventCreate(&start_event));
    CUDACHECK(cudaEventRecord(start_event, m_streams[current_comm]));
    if (torch::kFloat32 == send_tensor.dtype()) {
        NCCLCHECK(ncclAllGather(send_tensor.data_ptr<float>(), recv_tensor.data_ptr<float>(), send_tensor.numel(), ncclFloat, m_nccl_comms[current_comm], m_streams[current_comm]));
    } else if (torch::kInt64 == send_tensor.dtype()) {
        NCCLCHECK(ncclAllGather(send_tensor.data_ptr<long>(), recv_tensor.data_ptr<long>(), send_tensor.numel(), ncclInt64, m_nccl_comms[current_comm], m_streams[current_comm]));
    }
    cudaEvent_t event;
    CUDACHECK(cudaEventCreate(&event));
    CUDACHECK(cudaEventRecord(event, m_streams[current_comm]));
    m_op_start_events.push_back(start_event);
    m_op_events.push_back(event);
    int handler = static_cast<int>(m_op_events.size()) - 1;
    m_current_comm++;
    m_current_comm %= m_num_comms;
    m_sync_flags[current_comm] = false;
    return handler;
}

void Communicator::allReduceRB(torch::Tensor tensor) {
    int current_comm = m_current_comm;
    int root = 0;
    ncclDataType_t nccl_type = ncclFloat;
    NCCLCHECK(ncclReduce(tensor.data_ptr<float>(), tensor.data_ptr<float>(), tensor.numel(), ncclFloat, ncclSum, root, m_nccl_comms[current_comm], m_streams[current_comm]));
    NCCLCHECK(ncclBroadcast(tensor.data_ptr<float>(), tensor.data_ptr<float>(), tensor.numel(), nccl_type, root, m_nccl_comms[current_comm], m_streams[current_comm]));

    //CUDACHECK(cudaEventRecord(m_events[current_comm], m_streams[current_comm]));
    m_current_comm++;
    m_current_comm %= m_num_comms;
    m_sync_flags[current_comm] = false;
}
	
void Communicator::allReduceRSAG(torch::Tensor tensor) {
    int current_comm = m_current_comm;
    int n = tensor.numel();
    int p  = m_size;
    if (n < p) {
        NCCLCHECK(ncclAllReduce(tensor.data_ptr<float>(), tensor.data_ptr<float>(), tensor.numel(), ncclFloat, ncclSum, m_nccl_comms[m_current_comm], m_streams[m_current_comm]));
    } else {
        int n_per_worker = (n+p-1)/p;
        int padded_n = n_per_worker * p;

        auto options_float =
            torch::TensorOptions()
            .dtype(tensor.dtype())
            .device(tensor.device().type())
            .requires_grad(false);

        at::cuda::CUDAStream cuStream = m_torchstreams[current_comm];
        {
            at::cuda::CUDAStreamGuard guard(cuStream);
            torch::Tensor temp_result = torch::zeros(n_per_worker, options_float); 
            if (n < padded_n) {
                // should be padded to the multiple of p
                torch::Tensor padded_tensor = torch::zeros(padded_n, options_float); 
                padded_tensor.narrow(0, 0, n).copy_(tensor);
                NCCLCHECK(ncclReduceScatter(padded_tensor.data_ptr<float>(), temp_result.data_ptr<float>(), temp_result.numel(), ncclFloat, ncclSum, m_nccl_comms[current_comm], m_streams[current_comm]));
                NCCLCHECK(ncclAllGather(temp_result.data_ptr<float>(), padded_tensor.data_ptr<float>(), temp_result.numel(), ncclFloat, m_nccl_comms[m_current_comm], m_streams[m_current_comm]));
                tensor.copy_(padded_tensor.narrow(0, 0, n));
            } else {
                NCCLCHECK(ncclReduceScatter(tensor.data_ptr<float>(), temp_result.data_ptr<float>(), temp_result.numel(), ncclFloat, ncclSum, m_nccl_comms[current_comm], m_streams[current_comm]));
                NCCLCHECK(ncclAllGather(temp_result.data_ptr<float>(), tensor.data_ptr<float>(), temp_result.numel(), ncclFloat, m_nccl_comms[m_current_comm], m_streams[m_current_comm]));
            }
        }
    }

    //CUDACHECK(cudaEventRecord(m_events[m_current_comm], m_streams[m_current_comm]));
    m_current_comm++;
    m_current_comm %= m_num_comms;
}

void Communicator::allReduce(torch::Tensor tensor) {
    NCCLCHECK(ncclAllReduce(tensor.data_ptr<float>(), tensor.data_ptr<float>(), tensor.numel(), ncclFloat, ncclSum, m_nccl_comms[m_current_comm], m_streams[m_current_comm]));
    //CUDACHECK(cudaEventRecord(m_events[m_current_comm], m_streams[m_current_comm]));
    m_current_comm++;
    m_current_comm %= m_num_comms;
}

void Communicator::multiBcast(vector<torch::Tensor> &tensor_list, vector<torch::Tensor> &output_list, const std::function<void(torch::Tensor, torch::Tensor)> &op) {
    vector<int> tensor_ranks;
    int assigned_rank = 0;
    int num_comm_tensors = 0;
    int min_tensor_size = 512*512;
    for (unsigned i = 0; i < tensor_list.size(); i++) {
        torch::Tensor tensor = tensor_list[i];
        if (tensor.numel() < min_tensor_size) {
            tensor_ranks.push_back(-1);
        } else {
            tensor_ranks.push_back(assigned_rank);
            assigned_rank++;
            assigned_rank %= m_size;
            num_comm_tensors++;
        }
    }
    if (m_size > 1) {
        _extendComms(num_comm_tensors);
    }
	for (unsigned i = 0; i < tensor_list.size(); i++) {
        torch::Tensor tensor = tensor_list[i];
        torch::Tensor output = output_list[i];
        assigned_rank = tensor_ranks[i];
        if (assigned_rank == -1) {
            op(tensor, output);
        } else {
            if (assigned_rank == m_rank) {
                op(tensor, output);
            } 
        }
    }
	for (unsigned i = 0; i < tensor_list.size(); i++) {
        torch::Tensor output = output_list[i];
        assigned_rank = tensor_ranks[i];
        if (m_size > 1 and assigned_rank >= 0) {
            NCCLCHECK(ncclBroadcast(output.data_ptr<float>(), output.data_ptr<float>(), output.numel(), ncclFloat, assigned_rank, m_nccl_comms[m_current_comm], m_streams[m_current_comm])); 
            //CUDACHECK(cudaEventRecord(m_events[m_current_comm], m_streams[m_current_comm]));
            m_current_comm++;
            m_current_comm %= m_num_comms;
        }
    }
}

void Communicator::sendrecv(torch::Tensor send_tensor, torch::Tensor recv_tensor, int peer) {
    NCCLCHECK(ncclGroupStart());
    ncclDataType_t nccl_type;
    if (torch::kFloat32 == send_tensor.dtype()) {
        nccl_type = ncclFloat;
        NCCLCHECK(ncclSend(send_tensor.data_ptr<float>(), send_tensor.numel(), nccl_type, peer, m_nccl_comms[m_current_comm], m_streams[m_current_comm]));
        NCCLCHECK(ncclRecv(recv_tensor.data_ptr<float>(), recv_tensor.numel(), nccl_type, peer, m_nccl_comms[m_current_comm], m_streams[m_current_comm]));
    } else if (torch::kInt64 == send_tensor.dtype()) {
        nccl_type = ncclInt64;
        NCCLCHECK(ncclSend(send_tensor.data_ptr<long>(), send_tensor.numel(), nccl_type, peer, m_nccl_comms[m_current_comm], m_streams[m_current_comm]));
        NCCLCHECK(ncclRecv(recv_tensor.data_ptr<long>(), recv_tensor.numel(), nccl_type, peer, m_nccl_comms[m_current_comm], m_streams[m_current_comm]));
    }
    NCCLCHECK(ncclGroupEnd());
    //CUDACHECK(cudaEventRecord(m_events[m_current_comm], m_streams[m_current_comm]));

    m_current_comm++;
    m_current_comm %= m_num_comms;
}
