from typing import List

from langchain_core.pydantic_v1 import BaseModel, Field


class KeyHeadings(BaseModel):
    """Information about a property that is listed for sale"""

    # Note that all fields are required rather than optional!
    heading: str = Field(
        ...,
        description="The heading of the subsection of a terms and condition document",
    )
    description: str = Field(
        ..., description="Description of the heading and what it includes"
    )


class Response(BaseModel):
    """Final response to the question being asked"""

    key_headings: List[KeyHeadings] = Field(
        description="List of headings that can be used to generate a terms and conditions document"
    )
