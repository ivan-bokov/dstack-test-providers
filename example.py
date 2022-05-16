import os
import torch

if __name__ == '__main__':
    print("Start torch")
    world_size = int(os.environ['WORLD_SIZE'])
    rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])
    if rank != 0:
        os.environ['MASTER_ADDR'] = os.environ['MASTER_HOSTNAME']
        os.environ['MASTER_PORT'] = os.environ['MASTER_PORT_MAPPING_29500']
    master_addr = os.environ['MASTER_ADDR']
    master_port = os.environ['MASTER_PORT']
    print(f'[rank={rank}] start torch.distributed.init_process_group()')
    method = f'tcp://{master_addr}:{master_port}'
    #method = f'env://'
    print(method)
    torch.distributed.init_process_group(backend='nccl', init_method=method, world_size=world_size, rank=rank)
    print(f'[rank={rank}] finish torch.distributed.init_process_group()')
    device = torch.device('cpu')

    torch0 = torch.tensor(rank, device=device)
    torch.distributed.all_reduce(torch0, op=torch.distributed.ReduceOp.SUM)

    ret_ = sum(range(world_size))
    ret0 = torch0.detach().to('cpu').numpy()
    assert abs(ret_ - ret0) < 1e-4

    torch.distributed.destroy_process_group()
