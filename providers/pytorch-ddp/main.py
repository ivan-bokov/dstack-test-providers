from typing import List
import sys

from dstack import Provider, Job


class PytorchDDPProvider(Provider):
    def __init__(self):
        super().__init__(schema="providers/pytorch-ddp/schema.yaml")
        self.script = self.workflow.data["script"]
        self.requirements = self.workflow.data.get("requirements")
        self.environment = self.workflow.data.get("environment") or {}
        self.artifacts = self.workflow.data.get("artifacts")
        self.working_dir = self.workflow.data.get("working_dir")
        self.ports = self.workflow.data.get("ports") or []
        self.resources = self._resources()

    def _image(self):
        return "pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime"

    def _commands(self, node_rank):
        commands = ["printenv"]
        if self.requirements:
            commands.append("pip install -r " + self.requirements)
        environment_init = ""
        if self.environment:
            for name in self.environment:
                escaped_value = self.environment[name].replace('"', '\\"')
                environment_init += f"{name}=\"{escaped_value}\" "
        nproc = ""
        if self.resources.gpu:
            nproc = f"--nproc_per_node={self.resources.gpu}"
        nodes = self.workflow.data["resources"].get("nodes")
        commands.append(
            f"{environment_init}python -m torch.distributed.launch {nproc} --nnodes={nodes} --node_rank={node_rank} --master_addr=$MASTER_HOSTNAME --master_port=$MASTER_PORT_MAPPING_{self.ports[0]} {self.script}"
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
            ports=self.ports
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
