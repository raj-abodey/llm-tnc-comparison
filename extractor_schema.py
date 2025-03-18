from typing import List

from langchain_core.pydantic_v1 import BaseModel, Field


class KeyHeadings(BaseModel):
    """Information about the key headings found in a typical terms and conditions document"""

    # Note that all fields are required rather than optional!
    heading: str = Field(
        ...,
        description="The heading of the subsection of a terms and condition document",
    )
    description: str = Field(
        ..., description="Description of the heading and what it includes"
    )


class Response(BaseModel):
    """List of headings and description"""

    key_headings: List[KeyHeadings] = Field(
        description="List of headings that can be used to generate a terms and conditions document"
    )
