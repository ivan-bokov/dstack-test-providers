from typing import List

from dstack import Provider, Job


class DockerProvider(Provider):
    def __init__(self):
        super().__init__(schema="providers/pytorch-ddp/schema.yaml")
        self.image = self.workflow.data["image"]
        self.commands = self.workflow.data.get("commands")
        self.artifacts = self.workflow.data.get("artifacts")
        self.working_dir = self.workflow.data.get("working_dir")
        self.ports = self.workflow.data.get("ports") or []
        self.resources = self._resources()

    def create_jobs(self) -> List[Job]:
	    masterJob = Job(
            image_name=self.image,
            commands=self.commands,
            working_dir=self.working_dir,
            resources=self.resources,
            artifacts=self.artifacts,
            ports=self.ports
        )
        jobs = [masterJob]
        if self.resources.nodes>1:
            for i in range(self.resources.nodes-1):
                jobs.append(Job(
            	    image_name=self.image,
            	    commands=self.commands,
            	    working_dir=self.working_dir,
            	    resources=self.resources,
            	    artifacts=self.artifacts,
            	    master=masterJob
        	))
        return jobs


if __name__ == '__main__':
    provider = DockerProvider()
    provider.start()
