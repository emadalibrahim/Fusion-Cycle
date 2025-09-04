from pydantic import BaseModel, Field
from schemas.base import LowerCamelAliasModel
from wrappers import register_wrapper
from wrappers.base import BaseResponse, BaseWrapper


class FCInput(LowerCamelAliasModel):
    solute_smiles: str = Field(
        description="Solute SMILES for FC calculation. "
                    "Can be multi-molecule (separated by .), in which case "
                    "the max FC among the molecules will be returned",
        example="c1ccccc1"
    )
    solvent_smiles: str = Field(
        description="Solvent SMILES for FC calculation. "
                    "Can be multi-molecule (separated by .), in which case "
                    "the max FC among the molecules will be returned",
        example="CO"
    )
    temperature: float = Field(
        description="Temperature in [K] FC calculation. ",
        example="298.15"
    )
    density: float = Field(
        description="Density in [mol/L] FC calculation. ",
        example="12.5"
    )


class FCOutput(BaseModel):
    error: str
    status: str
    results: float


class FCResponse(BaseResponse):
    result: float | None


@register_wrapper(
    name="FC",
    input_class=FCInput,
    output_class=FCOutput,
    response_class=FCResponse
)
class FCWrapper(BaseWrapper):
    """Wrapper class for FC"""
    prefixes = ["FC"]

    def call_sync(self, input: FCInput) -> FCResponse:
        """
        Endpoint for synchronous call to FCr.
        """
        output = self.call_raw(input=input)
        response = self.convert_output_to_response(output)

        return response

    async def call_async(self, input: FCInput, priority: int = 0) -> str:
        """
        Endpoint for asynchronous call to FCr.
        """
        return await super().call_async(input=input, priority=priority)

    async def retrieve(self, task_id: str) -> FCResponse | None:
        return await super().retrieve(task_id=task_id)

    @staticmethod
    def convert_output_to_response(output: FCOutput
                                   ) -> FCResponse:
        if output.status == "SUCCESS":
            status_code = 200
            message = ""
            result = output.results
        else:
            status_code = 500
            message = f"Backend error encountered in FC " \
                      f"with the following error message {output.error}"
            result = None

        response = FCResponse(
            status_code=status_code,
            message=message,
            result=result
        )

        return response
