from inference.analyze import analyze_text

tests = [
    "Garlic cures cancer permanently without doctors",
    "Clinical studies show vaccination is safe",
    "Herbal tea eliminates diabetes in 5 days",
    "Doctors recommend medical supervision for treatment"
]

for t in tests:
    print("\nText:", t)
    print("Result:", analyze_text(t))
