defmodule A2aAgentWebWeb.SummarizerControllerTest do
  @moduledoc """
  Tests for the SummarizerController summarization endpoint.
  """
  use A2aAgentWebWeb.ConnCase, async: true

  @doc """
  POST /api/summarize returns a summary for customer feedback.
  """
  test "POST /api/summarize returns summary", %{conn: conn} do
    feedback = "The product is great but the delivery was slow."
    conn = post(conn, "/api/summarize", %{text: feedback})
    assert json_response(conn, 200)["summary"] =~ "product"
  end
end
