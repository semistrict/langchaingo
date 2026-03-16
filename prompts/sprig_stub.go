//go:build nosprig

package prompts

import "text/template"

func sprigFuncMap() template.FuncMap {
	return nil
}
