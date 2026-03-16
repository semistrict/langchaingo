//go:build nostarlark

package tools

import "errors"

func evaluateStarlark(_ string) (string, error) {
	return "", errors.New("starlark expression evaluation not available (built with nostarlark tag)")
}
