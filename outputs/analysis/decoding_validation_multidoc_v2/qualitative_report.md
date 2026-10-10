# Top-k / Top-p validation error analysis

Ranked by paired COMET score on identical Europarl multidoc v2 sentences. These rankings identify examples for human review; COMET does not establish that the preceding sentence caused an improvement.

## top_k

| Pair | N | Changed | Helped | Hurt | Tied | Mean delta (changed) |
|---|---:|---:|---:|---:|---:|---:|
| en-de | 100 | 19 | 9 | 10 | 0 | +0.0138 |
| en-fr | 50 | 9 | 5 | 4 | 0 | +0.0058 |
| All | 150 | 28 | 14 | 14 | 0 | +0.0113 |

### Most helped examples

**en-de / en/ep-01-03-15.xml / 1708** (COMET +0.4088)

- Previous: I feel I am in good company with Mr Staes, but also with our Vice-President, Mr Wiebenga, who is also committed to this region.
- Source: Commissioner, as far as this matter is concerned, you could make many people a lot happier this afternoon.
- Reference: Diesbezüglich können Sie, Herr Kommissar, heute Nachmittag viele Menschen in diesem Parlament doch ein klein wenig glücklicher machen.
- Beam: Herr Kommissar!
- Selected: Herr Kommissar, in Bezug auf diesen Punkt könnte man viele Menschen heute Nachmittag um einiges glücklich machen.

**en-fr / en/ep-05-06-23.xml / 513** (COMET +0.1926)

- Previous: You said earlier that you wanted Europe to do what it was set up to do.
- Source: Jean Monnet was the inspiration behind this and he wanted a system whereby, under the, the Community picked up power along the way.
- Reference: Jean Monnet est à l origine de ce principe et voulait un système dans lequel la Communauté recueille des pouvoirs au fil du temps dans le cadre de l …
- Beam: Jean Monnet a été l'inspiration qui l'a inspirée et il voulait un système permettant à la Communauté de prendre le pouvoir tout au long de la route.
- Selected: Jean Monnet est l'inspirateur à l'origine de la situation, et il voulait un système permettant à la Communauté de prendre le pouvoir.

**en-de / en/ep-10-05-18-017.xml / 169** (COMET +0.0943)

- Previous: (DE) Madam President, Commissioner Oettinger, as a Member of the Committee on the Environment, Public Health and Food Safety and as shadow rapporteur…
- Source: There is a lot of destruction of life that should be conserved.
- Reference: Hier wird vieles vernichtet, was am Leben zu erhalten wäre.
- Beam: Es gibt eine Menge von Zerstörungen, die man bewahren sollte.
- Selected: Es gibt viel zerstörtes Leben, das man bewahren sollte.

**en-fr / en/ep-08-01-14-018.xml / 46** (COMET +0.0569)

- Previous: The proposal before us is a good regulation and I hope that it will be understood at international level, so that it can also be properly applied.
- Source: Things that are not understood will of course not make much sense.
- Reference: Les choses qui ne sont pas comprises n'auront bien sûr pas beaucoup de sens.
- Beam: Les choses qui ne sont pas comprises ne feront évidemment pas grand-chose.
- Selected: Des choses qui ne sont pas comprises ne seront évidemment pas très pertinentes.

**en-fr / en/ep-10-05-18-017.xml / 204** (COMET +0.0566)

- Previous: I drive a lot, and I also fly a lot for business purposes.
- Source: At the weekend, hundreds of thousands of people will be flying south once again, and spending their Whit break in Mallorca.
- Reference: Ce week-end, des centaines de milliers de personnes emprunteront l'avion à destination du sud, afin de passer le week-end de la Pentecôte à Majorque.
- Beam: Au week-end, des centaines de milliers de personnes vont de nouveau voler vers le sud et passer leur vacances Whit à Mallorca.
- Selected: Au week-end, des centaines de milliers de personnes vont se rendre encore une fois au sud et passer leur temps libre à Majorque.

### Most hurt examples

**en-fr / en/ep-08-01-14-018.xml / 144** (COMET -0.1933)

- Previous: Yes, fortunately the regulation has come into force.
- Source: It worked: the Council treated the matter as urgent, Parliament treated the matter as urgent.
- Reference: Cela a fonctionné : le Conseil a traité la question en urgence, le Parlement a traité la question en urgence.
- Beam: Elle a fonctionné : le Conseil a considéré la question comme urgente, le Parlement comme urgente.
- Selected: Cela a fonctionné : le Conseil a jugé comme le dossier urgent, le Parlement comme urgent.

**en-de / en/ep-10-05-18-017.xml / 204** (COMET -0.1337)

- Previous: I drive a lot, and I also fly a lot for business purposes.
- Source: At the weekend, hundreds of thousands of people will be flying south once again, and spending their Whit break in Mallorca.
- Reference: Am Wochenende fliegen wieder Hunderttausende in den Süden, verbringen ihre Pfingstferien auf Mallorca.
- Beam: Am Wochenende werden Hunderte von Tausenden von Menschen wieder nach Süden fliegen und ihren Whit-Urlaub auf Mallorca verbringen.
- Selected: Am Wochenende fliegen Hunderte von Tausende noch einmal nach Süden und verbringen ihre Whit-Frühzeit auf Mallorca.

**en-de / en/ep-05-10-12.xml / 729** (COMET -0.0557)

- Previous: Our resolution is in the same vein as announcements during recent days from the Council and the Atomic Energy Agency.
- Source: These have emphasised political dialogue as the most appropriate instrument for convincing Iran.
- Reference: In diesen wurde der politische Dialog als das am besten geeignete Instrument hervorgehoben, um den Iran zu überzeugen.
- Beam: Sie haben den politischen Dialog als das geeignetste Instrument zur Überzeugung des Iran hervorgehoben.
- Selected: Sie haben den politischen Dialog als das geeignetste Instrument für die Überzeugung des Iran betont.

**en-de / en/ep-01-03-15.xml / 1019** (COMET -0.0532)

- Previous: Solutions must be sought through discussion, as has happened meanwhile in Senegal, the Ivory Coast and Uganda, where five pharmaceutical companies ar…
- Source: For that matter, the pharmaceutical industry often exaggerates its costs: after all huge profits are also made in this sector, especially on cosmetic…
- Reference: Übrigens, in der Kostenfrage übertreibt die Pharmaindustrie häufig: In diesem Sektor werden ja auch enorme Gewinne erzielt, vor allem bei Kosmetika u…
- Beam: Die pharmazeutische Industrie übertrieben oft ihre Kosten: Denn in diesem Sektor werden ja auch riesige Gewinne erzielt, vor allem auf Kosmetika und …
- Selected: Die Pharmaindustrie übertrieben oft ihre Kosten: Schließlich gehen in diesem Sektor riesigen Gewinnen zu, insbesondere bei den Kosmetik- und Präparat…

**en-fr / en/ep-03-01-30.xml / 557** (COMET -0.0513)

- Previous: The budget that we in the committee are now proposing is also too small, but we nonetheless see the fact that we have succeeded in increasing the bud…
- Source: This hopefully means that, together with the contributions from the EDF, slightly more than EUR 500 million will be invested in this area in the peri…
- Reference: Avec les subventions du FED, cela signifie que quelque 500 millions d'euros seront injectés dans ce secteur au cours des 4 prochaines années.
- Beam: Cela signifie, je l'espère, que, avec les contributions du FED, quelque 500 millions d'euros seront investis dans ce domaine au cours des quatre proc…
- Selected: Ceci signifie, peut-être, que, avec les contributions du FEDER, quelque peu plus de 500 millions d'euros seront investis dans ce domaine au cours des…

## top_p

| Pair | N | Changed | Helped | Hurt | Tied | Mean delta (changed) |
|---|---:|---:|---:|---:|---:|---:|
| en-de | 100 | 29 | 11 | 18 | 0 | +0.0067 |
| en-fr | 50 | 15 | 8 | 7 | 0 | +0.0090 |
| All | 150 | 44 | 19 | 25 | 0 | +0.0075 |

### Most helped examples

**en-de / en/ep-01-03-15.xml / 1708** (COMET +0.4130)

- Previous: I feel I am in good company with Mr Staes, but also with our Vice-President, Mr Wiebenga, who is also committed to this region.
- Source: Commissioner, as far as this matter is concerned, you could make many people a lot happier this afternoon.
- Reference: Diesbezüglich können Sie, Herr Kommissar, heute Nachmittag viele Menschen in diesem Parlament doch ein klein wenig glücklicher machen.
- Beam: Herr Kommissar!
- Selected: Herr Kommissar, in diesem Punkt könntet ihr viele Menschen heute Nachmittag noch viel glücklicher machen.

**en-fr / en/ep-08-05-19-027.xml / 187** (COMET +0.0756)

- Previous: I do not believe that anyone supports the introduction of a European system in the EU knowing all the drawbacks and disproportionate consequences it …
- Source: For now, we encourage other means of redress which are more rapid and at lower costs and leave the regulation to the Member States to be decided nati…
- Reference: À l'heure actuelle, nous préconisons d'autres moyens de recours plus rapides et plus économiques et nous suggérons de laisser les États membres le so…
- Beam: Pour l'instant, nous encourageons d'autres moyens de réparation plus rapides et à moindre coût et nous laissons le règlement aux États membres de déc…
- Selected: Pour l'instant, nous encourageons d'autres moyens de recours plus rapides et à moindre coût, et nous laissons la réglementation aux États membres pou…

**en-fr / en/ep-08-01-14-018.xml / 46** (COMET +0.0569)

- Previous: The proposal before us is a good regulation and I hope that it will be understood at international level, so that it can also be properly applied.
- Source: Things that are not understood will of course not make much sense.
- Reference: Les choses qui ne sont pas comprises n'auront bien sûr pas beaucoup de sens.
- Beam: Les choses qui ne sont pas comprises ne feront évidemment pas grand-chose.
- Selected: Des choses qui ne sont pas comprises ne seront évidemment pas très pertinentes.

**en-fr / en/ep-10-05-18-017.xml / 204** (COMET +0.0566)

- Previous: I drive a lot, and I also fly a lot for business purposes.
- Source: At the weekend, hundreds of thousands of people will be flying south once again, and spending their Whit break in Mallorca.
- Reference: Ce week-end, des centaines de milliers de personnes emprunteront l'avion à destination du sud, afin de passer le week-end de la Pentecôte à Majorque.
- Beam: Au week-end, des centaines de milliers de personnes vont de nouveau voler vers le sud et passer leur vacances Whit à Mallorca.
- Selected: Au week-end, des centaines de milliers de personnes vont se rendre encore une fois au sud et passer leur temps libre à Majorque.

**en-de / en/ep-08-01-14-018.xml / 32** (COMET +0.0518)

- Previous: This was proposed, while for developing countries especially it is important that there is complete transparency about what is being imported.
- Source: After all, we are talking about substances that are banned in the European Union or severely restricted.
- Reference: Schließlich sprechen wir hier von Stoffen, für die in der Europäischen Union Verbote oder strenge Beschränkungen gelten.
- Beam: Wir sprechen schließlich von Substanzen, die in der Europäischen Union verboten oder stark eingeschränkt sind.
- Selected: Immerhin sprechen wir hier von Substanzen, die in der Europäischen Union verboten oder streng eingeschränkt sind.

### Most hurt examples

**en-de / en/ep-10-05-18-017.xml / 169** (COMET -0.0610)

- Previous: (DE) Madam President, Commissioner Oettinger, as a Member of the Committee on the Environment, Public Health and Food Safety and as shadow rapporteur…
- Source: There is a lot of destruction of life that should be conserved.
- Reference: Hier wird vieles vernichtet, was am Leben zu erhalten wäre.
- Beam: Es gibt eine Menge von Zerstörungen, die man bewahren sollte.
- Selected: Es gibt viele Zerstörungen von Leben, die gewahrt werden sollten.

**en-de / en/ep-03-01-30.xml / 433** (COMET -0.0581)

- Previous: I would start by saying that I find it both important and remarkable that across party lines everybody is basically in agreement in describing the pr…
- Source: This is important, even though one Member did remark that I might not be the right Commissioner to deal with this debate considering the kind of prob…
- Reference: Dies ist wichtig, wenngleich von einem Abgeordneten bemerkt wurde, ich sei angesichts der Probleme, um die es hier geht, insbesondere der Handelsfrag…
- Beam: Das ist wichtig, obwohl ein Mitglied bemerkt hat, dass ich vielleicht nicht der richtige Kommissar sein könnte, um mit dieser Debatte umzugehen, wenn…
- Selected: Das ist wichtig, auch wenn ein Mitglied feststellen ließ, dass ich vielleicht nicht der richtige Kommissar bin, um diese Debatte mit den zu behandeln…

**en-de / en/ep-05-10-12.xml / 729** (COMET -0.0557)

- Previous: Our resolution is in the same vein as announcements during recent days from the Council and the Atomic Energy Agency.
- Source: These have emphasised political dialogue as the most appropriate instrument for convincing Iran.
- Reference: In diesen wurde der politische Dialog als das am besten geeignete Instrument hervorgehoben, um den Iran zu überzeugen.
- Beam: Sie haben den politischen Dialog als das geeignetste Instrument zur Überzeugung des Iran hervorgehoben.
- Selected: Sie haben den politischen Dialog als das geeignetste Instrument für die Überzeugung des Iran betont.

**en-de / en/ep-09-05-06-014.xml / 41** (COMET -0.0518)

- Previous: I would like to voice the hope that the agreed telecommunications package will eventually be passed, following further procedures.
- Source: Nevertheless, I am disappointed that, for the reasons mentioned by the honourable Member, the package has not been passed today.
- Reference: Nichtsdestotrotz bin ich aus den durch den Abgeordneten erwähnten Gründen darüber enttäuscht, dass das Paket heute nicht verabschiedet worden ist.
- Beam: Nichtsdestotrotz bin ich enttäuscht, dass das Paket aus den von der Abgeordneten erwähnten Gründen heute nicht verabschiedet wurde.
- Selected: Trotzdem bin ich enttäuscht, dass aus den von Frau Kollegin erwähnten Gründen das Paket heute nicht verabschiedet wurde.

**en-fr / en/ep-08-01-14-018.xml / 144** (COMET -0.0398)

- Previous: Yes, fortunately the regulation has come into force.
- Source: It worked: the Council treated the matter as urgent, Parliament treated the matter as urgent.
- Reference: Cela a fonctionné : le Conseil a traité la question en urgence, le Parlement a traité la question en urgence.
- Beam: Elle a fonctionné : le Conseil a considéré la question comme urgente, le Parlement comme urgente.
- Selected: Cela a fonctionné : le Conseil a jugé urgent, le Parlement l'a jugé urgent.

## Interpretation

Compare meaning and omissions manually in the source, reference and two outputs. Report both corpus BLEU/chrF and COMET; a positive COMET delta on one sentence is an inspection cue, not a significance test.
