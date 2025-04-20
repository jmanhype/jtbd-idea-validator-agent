defmodule A2aAgentWebWeb.PassThroughOperator do
  @moduledoc """
  Operator that simply returns its input unchanged.
  """
  @spec call(map()) :: map()
  def call(input), do: input
end
