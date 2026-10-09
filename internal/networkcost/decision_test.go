package networkcost_test

import (
	"testing"

	"goodkind.io/lm-semantic-search/internal/networkcost"
)

func TestDecideCoversEveryClassificationPreferenceAndOverride(t *testing.T) {
	t.Parallel()

	testCases := []struct {
		classification networkcost.Classification
		preference     networkcost.Preference
		want           networkcost.Decision
	}{
		{networkcost.ClassificationUnknown, networkcost.PreferenceAllow, networkcost.DecisionDownload},
		{networkcost.ClassificationNormal, networkcost.PreferenceAllow, networkcost.DecisionDownload},
		{networkcost.ClassificationExpensive, networkcost.PreferenceAllow, networkcost.DecisionDownload},
		{networkcost.ClassificationConstrained, networkcost.PreferenceAllow, networkcost.DecisionDownload},
		{networkcost.ClassificationExpensiveAndConstrained, networkcost.PreferenceAllow, networkcost.DecisionDownload},
		{networkcost.ClassificationUnknown, networkcost.PreferenceWarn, networkcost.DecisionDownloadWithWarning},
		{networkcost.ClassificationNormal, networkcost.PreferenceWarn, networkcost.DecisionDownload},
		{networkcost.ClassificationExpensive, networkcost.PreferenceWarn, networkcost.DecisionDownloadWithWarning},
		{networkcost.ClassificationConstrained, networkcost.PreferenceWarn, networkcost.DecisionDownloadWithWarning},
		{networkcost.ClassificationExpensiveAndConstrained, networkcost.PreferenceWarn, networkcost.DecisionDownloadWithWarning},
		{networkcost.ClassificationUnknown, networkcost.PreferenceDefer, networkcost.DecisionDefer},
		{networkcost.ClassificationNormal, networkcost.PreferenceDefer, networkcost.DecisionDownload},
		{networkcost.ClassificationExpensive, networkcost.PreferenceDefer, networkcost.DecisionDefer},
		{networkcost.ClassificationConstrained, networkcost.PreferenceDefer, networkcost.DecisionDefer},
		{networkcost.ClassificationExpensiveAndConstrained, networkcost.PreferenceDefer, networkcost.DecisionDefer},
	}

	for _, testCase := range testCases {
		name := string(testCase.classification) + "/" + string(testCase.preference)
		t.Run(name, func(t *testing.T) {
			t.Parallel()

			got := networkcost.Decide(testCase.classification, testCase.preference, false)
			if got != testCase.want {
				t.Errorf("Decide without override = %q want %q", got, testCase.want)
			}

			overridden := networkcost.Decide(testCase.classification, testCase.preference, true)
			if overridden != networkcost.DecisionDownload {
				t.Errorf(
					"Decide with override = %q want %q",
					overridden,
					networkcost.DecisionDownload,
				)
			}
		})
	}
}

func TestDefaultPreferenceWarnsOnUnknownNetwork(t *testing.T) {
	t.Parallel()

	got := networkcost.Decide(
		networkcost.ClassificationUnknown,
		networkcost.DefaultPreference,
		false,
	)
	if got != networkcost.DecisionDownloadWithWarning {
		t.Errorf("Decide = %q want %q", got, networkcost.DecisionDownloadWithWarning)
	}
}

func TestParsePreferenceAcceptsOnlyTheThreeValues(t *testing.T) {
	t.Parallel()

	accepted := map[string]networkcost.Preference{
		"allow":  networkcost.PreferenceAllow,
		"warn":   networkcost.PreferenceWarn,
		"defer":  networkcost.PreferenceDefer,
		" Warn ": networkcost.PreferenceWarn,
	}
	for value, want := range accepted {
		got, err := networkcost.ParsePreference(value)
		if err != nil {
			t.Errorf("ParsePreference(%q) returned error: %v", value, err)
		}
		if got != want {
			t.Errorf("ParsePreference(%q) = %q want %q", value, got, want)
		}
	}

	for _, value := range []string{"", "block", "unknown", "normal"} {
		got, err := networkcost.ParsePreference(value)
		if err == nil {
			t.Errorf("ParsePreference(%q) = %q, want an error", value, got)
		}
	}
}
