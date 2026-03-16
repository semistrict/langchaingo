//go:build !nostarlark

package chains

import (
	"strings"

	"go.starlark.net/lib/math"
	"go.starlark.net/starlark"
)

func (c LLMMathChain) evaluateExpression(expression string) (string, error) {
	expression = strings.TrimSpace(expression)
	v, err := starlark.Eval(&starlark.Thread{Name: "main"}, "input", expression, math.Module.Members)
	if err != nil {
		return "", err
	}
	return v.String(), nil
}
