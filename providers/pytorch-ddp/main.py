from typing import List, Optional
import sys

from dstack import Provider, Job, Resources, Gpu


class PytorchResources(Resources):
    def __init__(self, cpu: Optional[int] = None, memory: Optional[str] = None,
                 gpu: Optional[Gpu] = None, nodes: Optional[int] = 1):
        self.cpu = cpu
        self.memory = memory
        self.gpu = gpu
        self.nodes = nodes


class PytorchDDPProvider(Provider):
    def __init__(self):
        super().__init__(schema="providers/pytorch-ddp/schema.yaml")
        self.image = self.workflow.data["image"]
        self.commands = self.workflow.data.get("commands")
        self.artifacts = self.workflow.data.get("artifacts")
        self.working_dir = self.workflow.data.get("working_dir")
        self.ports = self.workflow.data.get("ports") or []
        self.resources = self.workflow.data.get("resources")

    def check_resoures(self) -> Optional[Resources]:
        if self.workflow.data.get("resources"):
            resources = Resources()
            if self.workflow.data["resources"].get("cpu"):
                if not str(self.workflow.data["resources"]["cpu"]).isnumeric():
                    sys.exit("resources.cpu in workflows.yaml should be an integer")
                cpu = int(self.workflow.data["resources"]["cpu"])
                if cpu > 0:
                    resources.cpu = cpu
            if self.workflow.data["resources"].get("nodes"):
                if not str(self.workflow.data["resources"]["nodes"]).isnumeric():
                    sys.exit("resources.nodes in workflows.yaml should be an integer")
                nodes = int(self.workflow.data["resources"]["nodes"])
                if nodes > 0:
                    resources.nodes = nodes
            if self.workflow.data["resources"].get("memory"):
                resources.memory = self.workflow.data["resources"]["memory"]
            if str(self.workflow.data["resources"].get("gpu")).isnumeric():
                gpu = int(self.workflow.data["resources"]["gpu"])
                if gpu > 0:
                    resources.gpu = Gpu(gpu)
            for resource_name in self.workflow.data["resources"]:
                if resource_name.endswith("/gpu") and len(resource_name) > 4:
                    if not str(self.workflow.data["resources"][resource_name]).isnumeric():
                        sys.exit(f"resources.'{resource_name}' in workflows.yaml should be an integer")
                    gpu = int(self.workflow.data["resources"][resource_name])
                    if gpu > 0:
                        resources.gpu = Gpu(gpu, name=resource_name[:-4])
            if resources.cpu or resources.memory or resources.gpu:
                return resources
            else:
                return None

    def create_jobs(self) -> List[Job]:
        masterJob = Job(
            image=self.image,
            commands=self.commands,
            working_dir=self.working_dir,
            resources=self.resources,
            artifacts=self.artifacts,
            ports=self.ports
        )
        jobs = [masterJob]
        if self.resources.nodes > 1:
            for i in range(self.resources.nodes - 1):
                jobs.append(Job(
                    image=self.image,
                    commands=self.commands,
                    working_dir=self.working_dir,
                    resources=self.resources,
                    artifacts=self.artifacts,
                    master=masterJob
                ))
        return jobs


if __name__ == '__main__':
    provider = PytorchDDPProvider()
    provider.start()
