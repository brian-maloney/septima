package septima

import "testing"

func TestPunctAgreementPick(t *testing.T) {
	cases := []struct {
		name string
		a, b string
		want int
	}{
		{"decimal recovered by crop", "5300", "53.00", pickSecond},
		{"decimal recovered by full", "168.61", "16861", pickFirst},
		{"colon recovered", "1217", "12:17", pickSecond},
		{"multi-row decimal", "2281\n228.5", "228.1\n228.5", pickSecond},
		{"malformed richer reading rejected", "108.00", "10.8.00", pickNone},
		{"double dot across rows still well-formed", "29.29\n13318", "29.29\n13.318", pickSecond},
		{"digit sequences differ", "5300", "53.10", pickNone},
		{"identical readings", "53.00", "53.00", pickNone},
		{"same punct count different placement", "5.300", "53.00", pickNone},
		{"empty candidate", "", "53.00", pickNone},
	}
	for _, c := range cases {
		if got := punctAgreementPick(c.a, c.b); got != c.want {
			t.Errorf("%s: punctAgreementPick(%q, %q) = %d, want %d", c.name, c.a, c.b, got, c.want)
		}
	}
}

func TestWellFormedRows(t *testing.T) {
	for s, want := range map[string]bool{
		"53.00":         true,
		"12:17":         true,
		"10.8.00":       false,
		"000.263.0":     false,
		"29.29\n13.318": true,
		"1.2\n3.4.5":    false,
	} {
		if got := wellFormedRows(s); got != want {
			t.Errorf("wellFormedRows(%q) = %v, want %v", s, got, want)
		}
	}
}
