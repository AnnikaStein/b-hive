from tasks.plotting import PlottingTask
from tasks.base import BaseTask
from rich.console import Console

c = Console()


class DeepJetRun(BaseTask):
    def requires(self):
        return PlottingTask.req(self)

    def output(self):
        return self.local_target("deepjetrun.txt")

    def run(self):
        c.print("Everything ready! Well done!")
