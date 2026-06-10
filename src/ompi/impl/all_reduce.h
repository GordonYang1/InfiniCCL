#ifndef INFINI_CCL_OMPI_IMPL_ALL_REDUCE_H_
#define INFINI_CCL_OMPI_IMPL_ALL_REDUCE_H_

#include <vector>

#include "base/all_reduce.h"
#include "communicator.h"
#include "dispatcher.h"
#include "logging.h"
#include "ompi/checks.h"
#include "ompi/comm_instance.h"
#include "ompi/half_precision.h"
#include "ompi/type_map.h"

namespace infini::ccl {

template <Device::Type device_type>
class AllReduceImpl<BackendType::kOmpi, device_type> {
 public:
  static ReturnStatus Apply(const void *send_buff, void *recv_buff,
                            size_t count, DataType data_type,
                            ReductionOpType op, Communicator *comm,
                            void *stream) {
    constexpr Device::Type kDev =
        ListGetBest<DevicePriority>(ActiveDevices<AllReduce>{});
    using Rt = Runtime<kDev>;

    auto *inst = static_cast<OmpiInstance *>(comm->inter_comm());

    if (!inst || inst->handle == MPI_COMM_NULL) {
      LOG("Invalid OpenMPI communicator instance for AllReduce.");
      return ReturnStatus::kInternalError;
    }

    // `kOmpiTypeMap` maps `kFloat16` / `kBFloat16` to `MPI_BYTE`, which MPI
    // cannot reduce (predefined reduction operations are undefined for
    // `MPI_BYTE`). Stage these payloads through host `float` buffers instead.
    if (data_type == DataType::kFloat16 || data_type == DataType::kBFloat16) {
      return ApplyHalfPrecision(send_buff, recv_buff, count, data_type, op,
                                comm, inst, stream);
    }

    MPI_Datatype mpi_type = DataTypeToOmpiType(data_type);
    MPI_Op mpi_op = RedOpToOmpiOp(op);
    size_t type_size = kDataTypeToSize.at(data_type);
    size_t total_bytes = count * type_size;

    // Handle GPU Memory (Staging Pattern)
    // Note: we simply use host-staging for now.
    void *host_sendbuf = malloc(total_bytes);
    void *host_recvbuf = malloc(total_bytes);
    if (!host_sendbuf || !host_recvbuf) {
      free(host_sendbuf);
      free(host_recvbuf);
      LOG("Failed to allocate host buffers for AllReduce staging.");
      return ReturnStatus::kSystemError;
    }

    // The send buffer may still be written by work queued on `stream`, so
    // synchronize before staging it to the host.
    CHECK_STATUS(Rt, Rt::StreamSynchronize(static_cast<Rt::Stream>(stream)));

    CHECK_STATUS(Rt, Rt::Memcpy(host_sendbuf, send_buff, total_bytes,
                                Rt::MemcpyDeviceToHost));

    INFINI_CHECK_MPI(MPI_Allreduce(host_sendbuf, host_recvbuf, count, mpi_type,
                                   mpi_op, inst->handle));

    if (op == ReductionOpType::kAvg) {
      int size = comm->size();
      float scale = 1.0f / static_cast<float>(size);

      DispatchFunc<kDev, AllTypes>(data_type, [&](auto dtype) {
        using T = typename decltype(dtype)::type;

        T *typed_buf = static_cast<T *>(host_recvbuf);

        // Simply do the averaging on the CPU before the H2D copy.
        for (size_t i = 0; i < count; ++i) {
          // TODO(lzm): should later use the unified `Cast` function instead of
          // static_cast to support CPU custom types.
          typed_buf[i] *= static_cast<T>(scale);
        }
      });
    }

    CHECK_STATUS(Rt, Rt::Memcpy(recv_buff, host_recvbuf, total_bytes,
                                Rt::MemcpyHostToDevice));

    free(host_sendbuf);
    free(host_recvbuf);

    return ReturnStatus::kSuccess;
  }

 private:
  // Reduces `kFloat16` / `kBFloat16` payloads by converting them to `float`
  // on the host, reducing with `MPI_FLOAT`, and converting the result back.
  static ReturnStatus ApplyHalfPrecision(const void *send_buff, void *recv_buff,
                                         size_t count, DataType data_type,
                                         ReductionOpType op, Communicator *comm,
                                         OmpiInstance *inst, void *stream) {
    constexpr Device::Type kDev =
        ListGetBest<DevicePriority>(ActiveDevices<AllReduce>{});
    using Rt = Runtime<kDev>;

    const size_t total_bytes = count * sizeof(uint16_t);
    std::vector<uint16_t> host_half(count);
    std::vector<float> host_send(count);
    std::vector<float> host_recv(count);

    // The send buffer may still be written by work queued on `stream`, so
    // synchronize before staging it to the host.
    CHECK_STATUS(Rt, Rt::StreamSynchronize(static_cast<Rt::Stream>(stream)));

    CHECK_STATUS(Rt, Rt::Memcpy(host_half.data(), send_buff, total_bytes,
                                Rt::MemcpyDeviceToHost));

    const bool is_bfloat16 = (data_type == DataType::kBFloat16);
    for (size_t i = 0; i < count; ++i) {
      host_send[i] = is_bfloat16 ? Bf16BitsToFloat(host_half[i])
                                 : Fp16BitsToFloat(host_half[i]);
    }

    INFINI_CHECK_MPI(MPI_Allreduce(host_send.data(), host_recv.data(), count,
                                   MPI_FLOAT, RedOpToOmpiOp(op), inst->handle));

    if (op == ReductionOpType::kAvg) {
      const float scale = 1.0f / static_cast<float>(comm->size());
      for (size_t i = 0; i < count; ++i) {
        host_recv[i] *= scale;
      }
    }

    for (size_t i = 0; i < count; ++i) {
      host_half[i] = is_bfloat16 ? FloatToBf16Bits(host_recv[i])
                                 : FloatToFp16Bits(host_recv[i]);
    }

    CHECK_STATUS(Rt, Rt::Memcpy(recv_buff, host_half.data(), total_bytes,
                                Rt::MemcpyHostToDevice));

    return ReturnStatus::kSuccess;
  }
};

template <>
struct BackendEnabled<AllReduce, BackendType::kOmpi> : std::true_type {};

}  // namespace infini::ccl

#endif  // INFINI_CCL_OMPI_IMPL_ALL_REDUCE_H_
