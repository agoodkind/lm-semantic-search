package semantic

import (
	"fmt"
	"strings"
)

// buildSearchFilter joins the extension filter and the relative-path prefix
// scope into one Milvus boolean expression, ANDing whichever clauses are
// present. An empty result means no filter, which searches the whole
// collection.
func buildSearchFilter(extensionFilter []string, relativePathPrefixes []string) string {
	clauses := make([]string, 0, 2)
	if extensionClause := buildExtensionFilter(extensionFilter); extensionClause != "" {
		clauses = append(clauses, extensionClause)
	}
	if prefixClause := buildRelativePathPrefixSetFilter(relativePathPrefixes); prefixClause != "" {
		clauses = append(clauses, prefixClause)
	}
	return strings.Join(clauses, " and ")
}

// buildRelativePathPrefixSetFilter ORs the per-prefix clauses so a search can
// scope to several subtrees at once, which is how a conversation-id set scopes
// retrieval to those conversations' rows. Empty or root prefixes contribute no
// clause; an empty result means the whole collection is searched.
func buildRelativePathPrefixSetFilter(relativePathPrefixes []string) string {
	clauses := make([]string, 0, len(relativePathPrefixes))
	for _, relativePathPrefix := range relativePathPrefixes {
		if clause := buildRelativePathPrefixFilter(relativePathPrefix); clause != "" {
			clauses = append(clauses, clause)
		}
	}
	if len(clauses) == 0 {
		return ""
	}
	if len(clauses) == 1 {
		return clauses[0]
	}
	return "(" + strings.Join(clauses, " or ") + ")"
}

// buildRelativePathPrefixFilter matches a directory and everything beneath it:
// the row whose relativePath equals the prefix, plus every row whose
// relativePath begins with the prefix and a separator. An empty or root prefix
// returns no clause so the whole collection is searched.
func buildRelativePathPrefixFilter(relativePathPrefix string) string {
	trimmed := strings.Trim(strings.TrimSpace(relativePathPrefix), "/")
	if trimmed == "" || trimmed == "." {
		return ""
	}
	return fmt.Sprintf(`(%s == "%s" or %s like "%s/%%")`, relativePathFieldName, escapeMilvusString(trimmed), relativePathFieldName, escapeMilvusLikePattern(trimmed))
}

// escapeMilvusLikePattern escapes a value for the literal portion of a Milvus
// LIKE pattern: backslash and double-quote for the surrounding string literal,
// plus the LIKE wildcards % and _ so they match literally. Without escaping the
// wildcards, a conversation id containing _ or % would over-match neighbors,
// which on the delete path could drop another conversation's rows. Backslash is
// escaped first so the escape characters added for the other cases survive.
func escapeMilvusLikePattern(value string) string {
	value = strings.ReplaceAll(value, `\`, `\\`)
	value = strings.ReplaceAll(value, `"`, `\"`)
	value = strings.ReplaceAll(value, "%", `\%`)
	value = strings.ReplaceAll(value, "_", `\_`)
	return value
}

func buildExtensionFilter(extensionFilter []string) string {
	cleanedExtensions := make([]string, 0, len(extensionFilter))
	for _, extension := range normalizeExtensionFilter(extensionFilter) {
		trimmedExtension := strings.TrimSpace(extension)
		if trimmedExtension == "" {
			continue
		}
		cleanedExtensions = append(cleanedExtensions, fmt.Sprintf("%q", trimmedExtension))
	}
	if len(cleanedExtensions) == 0 {
		return ""
	}
	return fileExtensionFieldName + " in [" + strings.Join(cleanedExtensions, ", ") + "]"
}
