//go:build nostarlark

package chains

import "errors"

func (c LLMMathChain) evaluateExpression(expression string) (string, error) {
	return "", errors.New("starlark expression evaluation not available (built with nostarlark tag)")
}
