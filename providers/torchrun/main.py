from typing import List, Optional
import sys
from argparse import ArgumentParser
import argparse

from dstack import Provider, Job, Resources, Gpu


class PytorchDDPProvider(Provider):
    def __init__(self):
        super().__init__(schema="providers/torchrun/schema.yaml")
        self.script = self.workflow.data.get("script") or self.workflow.data.get("file")
        self.version = str(self.workflow.data.get("version") or "3.9")
        self.requirements = self.workflow.data.get("requirements")
        self.environment = self.workflow.data.get("environment") or {}
        self.artifacts = self.workflow.data.get("artifacts")
        self.working_dir = self.workflow.data.get("working_dir")
        self.nodes = self.workflow.data.get("nodes") or 1
        self.resources = self._resources()
        self.args = self.workflow.data.get("args")

    def _resources(self) -> Optional[Resources]:
        resources = super()._resources()
        if resources.gpu is None:
            resources.gpu = Gpu(1)
        return resources

    def _image(self):
        return f"dstackai/python:{self.version}-cuda-11.1"

    def _commands(self, node_rank):
        commands = []
        if self.requirements:
            commands.append("pip3 install -r " + self.requirements)
        nproc = ""
        if self.resources.gpu:
            nproc = f"--nproc_per_node={self.resources.gpu.count}"
        args_init = ""
        if self.args:
            if isinstance(self.args, str):
                args_init += " " + self.args
            if isinstance(self.args, list):
                args_init += " " + ",".join(map(lambda arg: "\"" + arg.replace('"', '\\"') + "\"", self.args))
        torchrun_command = f"torchrun {nproc} --nnodes={self.nodes} --node_rank={node_rank}"
        if node_rank == 0:
            commands.append(
                f"{torchrun_command} --master_addr $JOB_HOSTNAME --master_port $JOB_PORT_0 {self.script} {args_init}"
            )
        else:
            commands.append(
                f"{torchrun_command} --master_addr $MASTER_JOB_HOSTNAME --master_port $MASTER_JOB_PORT_0 {self.script} {args_init}"
            )
        return commands

    def create_jobs(self) -> List[Job]:
        master_job = Job(
            image=self._image(),
            commands=self._commands(0),
            working_dir=self.working_dir,
            resources=self.resources,
            artifacts=self.artifacts,
            environment=self.environment,
            port_count=1,
        )
        jobs = [master_job]
        if self.nodes > 1:
            for i in range(self.nodes - 1):
                jobs.append(Job(
                    image=self._image(),
                    commands=self._commands(i+1),
                    working_dir=self.working_dir,
                    resources=self.resources,
                    environment=self.environment,
                    master=master_job
                ))
        return jobs

    def parse_args(self):
        parser = ArgumentParser(prog="dstack run torchrun")
        if not self.workflow.data.get("workflow_name"):
            parser.add_argument("file", metavar="FILE", type=str)
        parser.add_argument("-r", "--requirements", type=str, nargs="?")
        parser.add_argument('-e', '--env', action='append', nargs="?")
        parser.add_argument('--artifact', action='append', nargs="?")
        parser.add_argument("--working-dir", type=str, nargs="?")
        parser.add_argument("--cpu", type=int, nargs="?")
        parser.add_argument("--memory", type=str, nargs="?")
        parser.add_argument("--gpu", type=int, nargs="?")
        parser.add_argument("--gpu-name", type=str, nargs="?")
        parser.add_argument("--gpu-memory", type=str, nargs="?")
        parser.add_argument("--shm-size", type=str, nargs="?")
        parser.add_argument("--nnodes", type=int, nargs="?")
        if not self.workflow.data.get("workflow_name"):
            parser.add_argument("args", metavar="ARGS", nargs=argparse.ZERO_OR_MORE)
        args, unknown = parser.parse_known_args(self.provider_args)
        args.unknown = unknown
        if not self.workflow.data.get("workflow_name"):
            self.workflow.data["file"] = args.file
            _args = args.unknown + args.args
            if _args:
                self.workflow.data["args"] = _args
        if args.requirements:
            self.workflow.data["requirements"] = args.requirements
        if args.artifact:
            self.workflow.data["artifacts"] = args.artifact
        if args.working_dir:
            self.workflow.data["working_dir"] = args.working_dir
        if args.env:
            environment = self.workflow.data.get("environment") or {}
            for e in args.env:
                if "=" in e:
                    tokens = e.split("=", maxsplit=1)
                    environment[tokens[0]] = tokens[1]
                else:
                    environment[e] = ""
            self.workflow.data["environment"] = environment
        if args.nnodes:
            self.workflow.data["nodes"] = args.nnodes
        if args.cpu or args.memory or args.gpu or args.gpu_name or args.gpu_memory or args.shm_size:
            resources = self.workflow.data.get("resources") or {}
            self.workflow.data["resources"] = resources
            if args.cpu:
                resources["cpu"] = args.cpu
            if args.memory:
                resources["memory"] = args.memory
            if args.gpu or args.gpu_name or args.gpu_memory:
                gpu = self.workflow.data["resources"].get("gpu") or {} if self.workflow.data.get("resources") else {}
                if type(gpu) is int:
                    gpu = {
                        "count": gpu
                    }
                resources["gpu"] = gpu
                if args.gpu:
                    gpu["count"] = args.gpu
                if args.gpu_memory:
                    gpu["memory"] = args.gpu_memory
                if args.gpu_name:
                    gpu["name"] = args.gpu_name
            if args.shm_size:
                resources["shm_size"] = args.shm_size


if __name__ == '__main__':
    provider = PytorchDDPProvider()
    provider.start()
