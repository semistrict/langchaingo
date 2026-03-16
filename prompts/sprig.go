//go:build !nosprig

package prompts

import (
	"text/template"

	"github.com/Masterminds/sprig/v3"
)

func sprigFuncMap() template.FuncMap {
	return sprig.TxtFuncMap()
}
