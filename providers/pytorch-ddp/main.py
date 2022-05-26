from typing import List
import sys

from dstack import Provider, Job


class PytorchDDPProvider(Provider):
    def __init__(self):
        super().__init__(schema="providers/pytorch-ddp/schema.yaml")
        self.script = self.workflow.data["script"]
        self.version = str(self.workflow.data.get("version") or "3.9")
        self.requirements = self.workflow.data.get("requirements")
        self.environment = self.workflow.data.get("environment") or {}
        self.artifacts = self.workflow.data.get("artifacts")
        self.working_dir = self.workflow.data.get("working_dir")
        self.resources = self._resources()

    def _image(self):
        cuda_is_required = self.resources and self.resources.gpu
        return f"dstackai/python:{self.version}-cuda-11.1.1" if cuda_is_required else f"python:{self.version}"
        #return "pytorch/pytorch:1.9.1-cuda11.1-cudnn8-runtime"

    def _commands(self, node_rank):
        commands = ["printenv", "echo $MASTER_HOSTNAME"]
        if self.requirements:
            commands.append("pip3 install -r " + self.requirements)
        if self.environment:
            for name in self.environment:
                escaped_value = self.environment[name].replace('"', '\\"')
                commands.append(f"export {name}=\"{escaped_value}\"")
        nproc = ""
        if self.resources.gpu:
            nproc = f"--nproc_per_node={self.resources.gpu.count}"
        nodes = self.workflow.data["resources"].get("nodes")
        if node_rank == 0:
            commands.append(
                f"torchrun {nproc} --max_restarts=3 --nnodes=1:{nodes} --node_rank={node_rank} --rdzv_id=$MASTER_JOB_ID --rdzv_backend=c10d --master_addr $MASTER_HOSTNAME --master_port $MASTER_PORT_0 {self.script}"
            )
        else:
            commands.append(
                f"torchrun {nproc} --max_restarts=3 --nnodes=1:{nodes} --node_rank={node_rank} --rdzv_id=$MASTER_JOB_ID --rdzv_backend=c10d --rdzv_endpoint=$MASTER_HOST_NODE_ADDR_0 --master_addr $MASTER_HOSTNAME --master_port $MASTER_PORT_0 {self.script}"
                # --master_addr $MASTER_HOSTNAME --master_port $MASTER_PORT_0 {self.script}"
            )
        return commands

    def create_jobs(self) -> List[Job]:
        nodes = 1
        if self.workflow.data["resources"].get("nodes"):
            if not str(self.workflow.data["resources"]["nodes"]).isnumeric():
                sys.exit("resources.nodes in workflows.yaml should be an integer")
            if int(self.workflow.data["resources"]["nodes"]) > 1:
                nodes = int(self.workflow.data["resources"]["nodes"])
        masterJob = Job(
            image=self._image(),
            commands=self._commands(0),
            working_dir=self.working_dir,
            resources=self.resources,
            artifacts=self.artifacts,
            ports=[29500]
        )
        jobs = [masterJob]
        if nodes > 1:
            for i in range(nodes - 1):
                jobs.append(Job(
                    image=self._image(),
                    commands=self._commands(i+1),
                    working_dir=self.working_dir,
                    resources=self.resources,
                    artifacts=self.artifacts,
                    master=masterJob
                ))
        return jobs


if __name__ == '__main__':
    provider = PytorchDDPProvider()
    provider.start()
