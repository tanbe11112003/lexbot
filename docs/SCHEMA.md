# Graph Schema

## Quan hệ đúng theo mô hình đề xuất

```text
(:Law)-[:HAS_PART]->(:Part)
(:Part)-[:HAS_CHAPTER]->(:Chapter)
(:Chapter)-[:HAS_ARTICLE]->(:Article)
(:Chapter)-[:HAS_SECTION]->(:Section)-[:HAS_ARTICLE]->(:Article)
(:Article)-[:DEFINES_CRIME]->(:Crime)
(:Article)-[:HAS_CLAUSE]->(:Clause)
(:Clause)-[:HAS_POINT]->(:Point)
(:Clause)-[:HAS_CONDITION]->(:Condition)
(:Point)-[:HAS_CONDITION]->(:Condition)
(:Clause)-[:HAS_PENALTY_FRAME]->(:PenaltyFrame)
(:Point)-[:HAS_PENALTY_FRAME]->(:PenaltyFrame)
(:PenaltyFrame)-[:HAS_MAIN_PENALTY]->(:Penalty)
(:PenaltyFrame)-[:HAS_ADDITIONAL_PENALTY]->(:Penalty)
(:Article)-[:REFERENCES]->(:Article)
(:Crime)-[:HAS_ACT_REQUIREMENT]->(:ActRequirement)
(:Crime)-[:HAS_SUBJECT_REQUIREMENT]->(:SubjectRequirement)
(:Crime)-[:HAS_OBJECT_REQUIREMENT]->(:ObjectRequirement)
(:Crime)-[:HAS_CONSEQUENCE_REQUIREMENT]->(:ConsequenceRequirement)
(:Crime)-[:HAS_QUANTITY_THRESHOLD]->(:QuantityThreshold)
(:Article)-[:HAS_EXCEPTION]->(:Exception)
(:Article)-[:HAS_JUDICIAL_MEASURE]->(:JudicialMeasure)
```

## Quan hệ mở rộng cho chatbot

```text
(:Article)-[:HAS_RULE]->(:Rule)
(:Clause)-[:REPRESENTS_RULE]->(:Rule)
(:Point)-[:REPRESENTS_RULE]->(:Rule)
(:Article)-[:HAS_LEGAL_CONCEPT]->(:LegalConcept)
(:Article)-[:HAS_MITIGATING_FACTOR]->(:MitigatingFactor)
(:Article)-[:HAS_AGGRAVATING_FACTOR]->(:AggravatingFactor)
(:SlangTerm)-[:NORMALIZES_TO]->(:LegalConcept|Substance)
(:SlangTerm)-[:MAY_INDICATE]->(:LegalSignal)
(:ActionAlias)-[:MAY_INDICATE]->(:LegalSignal)
(:SubstanceAlias)-[:NORMALIZES_TO]->(:Substance)
(:LegalSignal)-[:RELATED_TO]->(:Article)
```
