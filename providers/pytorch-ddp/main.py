from typing import List
import sys

from dstack import Provider, Job


class PytorchDDPProvider(Provider):
    def __init__(self):
        super().__init__(schema="providers/pytorch-ddp/schema.yaml")
        self.image = self.workflow.data["image"]
        self.commands = self.workflow.data.get("commands")
        self.artifacts = self.workflow.data.get("artifacts")
        self.working_dir = self.workflow.data.get("working_dir")
        self.ports = self.workflow.data.get("ports") or []
        self.resources = self._resources()

    def create_jobs(self) -> List[Job]:
        nodes = 1
        if self.workflow.data["resources"].get("nodes"):
            if not str(self.workflow.data["resources"]["nodes"]).isnumeric():
                sys.exit("resources.nodes in workflows.yaml should be an integer")
            if int(self.workflow.data["resources"]["nodes"]) > 1:
                nodes = int(self.workflow.data["resources"]["nodes"])
        masterJob = Job(
            image=self.image,
            commands=self.commands,
            working_dir=self.working_dir,
            resources=self.resources,
            artifacts=self.artifacts,
            ports=self.ports
        )
        jobs = [masterJob]
        if nodes > 1:
            for i in range(nodes - 1):
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
