//go:build live && production

package live

import (
	"fmt"
	"os"
	"testing"
)

const (
	productionDatabaseName     = "default"
	productionOptInEnvironment = "LMS_PRODUCTION_CONFIRM_DATABASE"
)

func validateProductionOptIn(value string) error {
	if value != productionDatabaseName {
		return fmt.Errorf(
			"%s must equal %q to run production validation",
			productionOptInEnvironment,
			productionDatabaseName,
		)
	}
	return nil
}

func requireProductionOptIn(t *testing.T) {
	t.Helper()
	if err := validateProductionOptIn(os.Getenv(productionOptInEnvironment)); err != nil {
		t.Fatal(err)
	}
}

func TestValidateProductionOptIn(t *testing.T) {
	t.Parallel()
	for _, testCase := range []struct {
		name    string
		value   string
		wantErr bool
	}{
		{name: "missing", wantErr: true},
		{name: "wrong database", value: "sandbox", wantErr: true},
		{name: "default database", value: "default"},
	} {
		t.Run(testCase.name, func(t *testing.T) {
			t.Parallel()
			err := validateProductionOptIn(testCase.value)
			if (err != nil) != testCase.wantErr {
				t.Fatalf("validateProductionOptIn(%q) error = %v, want error %v", testCase.value, err, testCase.wantErr)
			}
		})
	}
}
