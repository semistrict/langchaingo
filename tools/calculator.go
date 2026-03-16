package tools

import (
	"context"

	"github.com/tmc/langchaingo/callbacks"
)

// Calculator is a tool that can do math.
type Calculator struct {
	CallbacksHandler callbacks.Handler
}

var _ Tool = Calculator{}

// Description returns a string describing the calculator tool.
func (c Calculator) Description() string {
	return `Useful for getting the result of a math expression.
	The input to this tool should be a valid mathematical expression that could be executed by a starlark evaluator.`
}

// Name returns the name of the tool.
func (c Calculator) Name() string {
	return "calculator"
}

// Call evaluates the input using a starlak evaluator and returns the result as a
// string. If the evaluator errors the error is given in the result to give the
// agent the ability to retry.
func (c Calculator) Call(ctx context.Context, input string) (string, error) {
	if c.CallbacksHandler != nil {
		c.CallbacksHandler.HandleToolStart(ctx, input)
	}

	result, err := evaluateStarlark(input)
	if err != nil {
		return err.Error(), nil //nolint:nilerr
	}

	if c.CallbacksHandler != nil {
		c.CallbacksHandler.HandleToolEnd(ctx, result)
	}

	return result, nil
}
