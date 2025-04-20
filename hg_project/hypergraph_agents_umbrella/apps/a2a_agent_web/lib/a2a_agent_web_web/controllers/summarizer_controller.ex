defmodule A2aAgentWebWeb.SummarizerController do
  @moduledoc """
  Controller for summarizing customer feedback using OpenAI.
  """
  use A2aAgentWebWeb, :controller

  @doc """
  POST /api/summarize
  Receives JSON {"text": ...} and returns a summary.
  """
  @spec summarize(Plug.Conn.t(), map()) :: Plug.Conn.t()
  def summarize(conn, %{"text" => text}) do
    graph = %{summarize: %{operator: A2aAgentWebWeb.PassThroughOperator, deps: []}}
    agent_map = %{summarize: A2aAgentWebWeb.MyOpenAISummarizerAgent}
    input = %{"text" => text}

    case HypergraphAgent.BasicOrchestrator.orchestrate(graph, agent_map, input) do
      {:ok, %{:summarize => %{"summary" => summary}}} ->
        json(conn, %{"summary" => summary})
      {:ok, %{"summarize" => %{"summary" => summary}}} ->
        json(conn, %{"summary" => summary})
      {:error, reason} ->
        conn |> put_status(500) |> json(%{"error" => inspect(reason)})
    end
  end
end
