//go:build !nostarlark

package tools

import (
	"fmt"

	"go.starlark.net/lib/math"
	"go.starlark.net/starlark"
)

func evaluateStarlark(input string) (string, error) {
	v, err := starlark.Eval(&starlark.Thread{Name: "main"}, "input", input, math.Module.Members)
	if err != nil {
		return "", fmt.Errorf("error from evaluator: %w", err)
	}
	return v.String(), nil
}
