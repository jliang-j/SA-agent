from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator


Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class Task(BaseModel):
    """研究工具接收的独立子任务，不包含任意工具或依赖声明。"""

    model_config = ConfigDict(extra="forbid")
    id: Annotated[Text, Field(pattern=r"^t[1-9][0-9]*$")]
    goal: Text
    query: Text


class Plan(BaseModel):
    """一次研究调用的任务集合，最多三个任务；补查限制由工具另行检查。"""

    model_config = ConfigDict(extra="forbid")
    tasks: list[Task] = Field(min_length=1, max_length=3)

    @model_validator(mode="after")
    def unique_ids(self):
        """拒绝同一计划中的重复任务编号。"""
        ids = [task.id for task in self.tasks]
        if len(ids) != len(set(ids)):
            raise ValueError("任务编号不能重复")
        return self
