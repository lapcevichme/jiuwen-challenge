import asyncio
from collections.abc import Mapping

from openjiuwen.core.session.config.base import WORKFLOW_EXECUTE_TIMEOUT

from openjiuwen.core.workflow import (
    End,
    Start,
    Workflow,
    WorkflowCard,
    WorkflowComponent,
    WorkflowExecutionState,
    create_workflow_session,
)

from harness.contracts import SolveResult, SolveStatus, Solver, TaskSpec


class _SolveComponent(WorkflowComponent):
    def __init__(self, task: TaskSpec, solver: Solver):
        super().__init__()
        self.task = task
        self.solver = solver

    async def invoke(self, inputs, session, context):
        result = await asyncio.to_thread(self.solver, self.task)
        return {"result": {
            "status": result.status.value,
            "answer": result.answer,
            "detail": result.detail,
        }}


class SolverWorkflow:
    def __init__(self, registry: Mapping[str, Solver]):
        self.registry = dict(registry)

    async def solve(self, task: TaskSpec) -> SolveResult:
        solver = self.registry.get(task.task_type)
        if solver is None:
            return SolveResult.unsupported(f"no solver for task type {task.task_type!r}")
        flow = Workflow(card=WorkflowCard(id="contest-solver", name="Contest solver", version="1"))
        flow.set_start_comp("start", Start())
        flow.add_workflow_comp("solve", _SolveComponent(task, solver))
        flow.set_end_comp("end", End(), inputs_schema={"result": "${solve.result}"})
        flow.add_connection("start", "solve")
        flow.add_connection("solve", "end")
        output = await flow.invoke({}, session=create_workflow_session(envs={WORKFLOW_EXECUTE_TIMEOUT: 75}))
        if output.state is not WorkflowExecutionState.COMPLETED:
            return SolveResult.error(f"workflow ended with state {output.state.value}")
        result = output.result["output"]["result"]
        return SolveResult(SolveStatus(result["status"]), result.get("answer"), result["detail"])
