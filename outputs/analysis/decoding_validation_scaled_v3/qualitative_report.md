# Top-k / Top-p validation error analysis

Ranked by paired COMET score on identical Europarl scaled_v3 sentences. These rankings identify examples for human review; COMET does not establish that the preceding sentence caused an improvement.

## top_k

| Pair | N | Changed | Helped | Hurt | Tied | Mean delta (changed) |
|---|---:|---:|---:|---:|---:|---:|
| en-de | 1000 | 173 | 77 | 96 | 0 | +0.0015 |
| en-fr | 500 | 117 | 61 | 56 | 0 | +0.0044 |
| All | 1500 | 290 | 138 | 152 | 0 | +0.0027 |

### Most helped examples

**en-fr / en/ep-98-07-15.xml / 3551** (COMET +0.2997)

- Previous: Where it is a question of the freedom of movement of people, of the social rules and measures involved, we obstruct it.
- Source: Where any kind of discriminatory rules are concerned, we disgrace ourselves.
- Reference: Lorsqu'il s'agit de règles discriminatoires, nous avons honte.
- Beam: En ce qui concerne tout type de règles discriminatoires, nous nous reprochons.
- Selected: Nous nous rendons honteux lorsqu'il s'agit de la réglementation discriminatoire.

**en-de / en/ep-97-02-18.xml / 37** (COMET +0.2402)

- Previous: I did nothing of the sort.
- Source: It misleads the media and the outside world.
- Reference: Es handelt sich hier um eine Irreführung der Medien und der Öffentlichkeit außerhalb dieses Hauses.
- Beam: Sie misst die Medien und die Außenwelt.
- Selected: Es ist die Medien und die Außenwelt irreführend.

**en-de / en/ep-09-09-17-002.xml / 236** (COMET +0.2153)

- Previous: My Eastern European colleagues and I on the Committee on Agriculture and Rural Development warned on several occasions that we needed a much greater …
- Source: Unfortunately, we have remained a lone voice in the desert, and the fault for this lies with Parliament.
- Reference: Leider wollte aber keiner auf uns hören, und die Schuld dafür ist dem Parlament zuzuschreiben.
- Beam: Leider sind wir in der Wüste eine einzige Stimme geblieben, und dafür liegt das Parlament schuldig.
- Selected: Leider sind wir in der Wüste allein geblieben, und die Schuld dafür liegt bei dem Parlament.

**en-de / en/ep-06-10-24-004.xml / 43** (COMET +0.2076)

- Previous: What began with the first mass demonstration in Budapest on 23 October 1956 and ended when Soviet troops were sent in on 4 November was a revolution …
- Source: The participants in the Hungarian uprising quite rightly spoke of revolution, or forradalom.
- Reference: Zu Recht sprachen schon die Beteiligten des ungarischen Volksaufstandes von der Revolution, von forradalom.
- Beam: Die Teilnehmer des ungarischen Aufstands sprachen zu Recht von Revolution, also "Revolution".
- Selected: Die Teilnehmer der ungarischen Aufstandsbewegung haben zu Recht von Revolution oder Volksrevolution gesprochen.

**en-fr / en/ep-09-11-24-003.xml / 238** (COMET +0.1964)

- Previous: I would very much like to do that, but it requires a global agreement.
- Source: Otherwise, this extra 10% from the EU would be eaten up by only two years of emissions increases in China, and we will still not have saved the clima…
- Reference: Autrement, ces 10 % supplémentaires de la part de l'UE seront anéantis par seulement deux années d'augmentation des émissions en Chine et nous n'auro…
- Beam: Sinon, ces 10 % supplémentaires de l'UE seraient ennuyés par deux années seulement d'augmentations des émissions en Chine, et nous n'aurons toujours …
- Selected: Sinon, ce 10 % supplémentaire provenant de l'UE serait épuisé par deux ans seulement d'augmentations des émissions en Chine, et nous n'aurons toujour…

### Most hurt examples

**en-fr / en/ep-07-09-25-012.xml / 106** (COMET -0.6051)

- Previous: You are therefore considering whether you should say that at least 50% must in future go to these alternatives, because my home city of Berlin argues…
- Source: I think that is mad!
- Reference: Je trouve cela aberrant !
- Beam: Je pense que c'est folle !
- Selected: Je pense que c'est de l'aile !

**en-de / en/ep-03-04-08.xml / 3008** (COMET -0.2306)

- Previous: The existing control arrangements with regard to data protection that it is proposed be continued are a parody of the rule of law.
- Source: As a Northerner, I say that we can do better ourselves, at least while we have not yet squandered the whole of our heritage in terms of legal culture.
- Reference: Als Nordeuropäer sage ich, dass wir das selbst besser erledigen können, jedenfalls solange wir noch nicht unser gesamtes rechtskulturelles Erbe preis…
- Beam: Als Nordamerikaner sage ich, dass wir uns selbst besser machen können, zumindest, wenn wir noch nicht das gesamte Erbe in Bezug auf die Rechtskultur …
- Selected: Als Norddörfer sage ich, dass wir uns besser machen können, zumindest wenn wir noch nicht das gesamte Erbe uns als Rechtskultur verschwendet haben.

**en-de / en/ep-06-11-14-005.xml / 250** (COMET -0.2178)

- Previous: (HU) Madam President, the report by the European Court of Auditors, which, as in previous years, is the result of very thorough, high-quality work, s…
- Source: I wish to emphasise the significance of this statement because once again this year, as in the past 12 years, the Court is unable to give an unqualif…
- Reference: Ich möchte auf die Bedeutung dieser Erklärung hinweisen, denn auch in diesem Jahr - wie in den vergangenen 12 Jahren - kann der Gerichtshof keine une…
- Beam: Ich möchte die Bedeutung dieser Erklärung unterstreichen, weil der Rechnungshof in diesem Jahr wie in den letzten 12 Jahren wieder einmal unqualifizi…
- Selected: Ich möchte die Bedeutung dieses Statements hervorheben, weil der Rechnungshof sich in diesem Jahr wieder einmal, wie in den vergangenen zwölf Jahren,…

**en-de / en/ep-02-09-05.xml / 1408** (COMET -0.1962)

- Previous: The people must know that we understand their desire to acquire land and that we want to provide financial support for a legal procedure for acquirin…
- Source: We must help Zimbabwe's neighbours to take care of refugees.
- Reference: Wir müssen den Nachbarländern bei der Aufnahme von Flüchtlingen helfen.
- Beam: Wir müssen den Nachbarn von Simbabwe helfen, die Flüchtlinge zu versorgen.
- Selected: Wir müssen den Nachbarn von Simbabwe helfen, um Flüchtlingen Sorge zu tragen.

**en-de / en/ep-05-06-08.xml / 1039** (COMET -0.1601)

- Previous: & # 160; & # 160; Mr President, this concerns the way in which we interpret the Rules of Procedure.
- Source: I arranged with the Secretary-General to have a one-minute explanation.
- Reference: Ich habe mit dem Generalsekretär eine Redezeit von einer Minute vereinbart.
- Beam: Ich habe mit dem Generalsekretär eine einminütige Erklärung vereinbart.
- Selected: Ich habe mit dem Generalsekretär eine Wortmeldung von einer Minute getroffen.

## top_p

| Pair | N | Changed | Helped | Hurt | Tied | Mean delta (changed) |
|---|---:|---:|---:|---:|---:|---:|
| en-de | 1000 | 270 | 124 | 146 | 0 | +0.0005 |
| en-fr | 500 | 151 | 78 | 73 | 0 | -0.0019 |
| All | 1500 | 421 | 202 | 219 | 0 | -0.0003 |

### Most helped examples

**en-fr / en/ep-98-07-15.xml / 3551** (COMET +0.3071)

- Previous: Where it is a question of the freedom of movement of people, of the social rules and measures involved, we obstruct it.
- Source: Where any kind of discriminatory rules are concerned, we disgrace ourselves.
- Reference: Lorsqu'il s'agit de règles discriminatoires, nous avons honte.
- Beam: En ce qui concerne tout type de règles discriminatoires, nous nous reprochons.
- Selected: Nous nous rendons honteux lorsqu'il s'agit de tout type de règles discriminatoires.

**en-de / en/ep-97-02-18.xml / 37** (COMET +0.2402)

- Previous: I did nothing of the sort.
- Source: It misleads the media and the outside world.
- Reference: Es handelt sich hier um eine Irreführung der Medien und der Öffentlichkeit außerhalb dieses Hauses.
- Beam: Sie misst die Medien und die Außenwelt.
- Selected: Es ist die Medien und die Außenwelt irreführend.

**en-de / en/ep-09-09-17-002.xml / 236** (COMET +0.2153)

- Previous: My Eastern European colleagues and I on the Committee on Agriculture and Rural Development warned on several occasions that we needed a much greater …
- Source: Unfortunately, we have remained a lone voice in the desert, and the fault for this lies with Parliament.
- Reference: Leider wollte aber keiner auf uns hören, und die Schuld dafür ist dem Parlament zuzuschreiben.
- Beam: Leider sind wir in der Wüste eine einzige Stimme geblieben, und dafür liegt das Parlament schuldig.
- Selected: Leider sind wir in der Wüste allein geblieben, und die Schuld dafür liegt bei dem Parlament.

**en-de / en/ep-06-10-24-004.xml / 43** (COMET +0.2076)

- Previous: What began with the first mass demonstration in Budapest on 23 October 1956 and ended when Soviet troops were sent in on 4 November was a revolution …
- Source: The participants in the Hungarian uprising quite rightly spoke of revolution, or forradalom.
- Reference: Zu Recht sprachen schon die Beteiligten des ungarischen Volksaufstandes von der Revolution, von forradalom.
- Beam: Die Teilnehmer des ungarischen Aufstands sprachen zu Recht von Revolution, also "Revolution".
- Selected: Die Teilnehmer der ungarischen Aufstandsbewegung haben zu Recht von Revolution oder Volksrevolution gesprochen.

**en-fr / en/ep-09-11-24-003.xml / 238** (COMET +0.1964)

- Previous: I would very much like to do that, but it requires a global agreement.
- Source: Otherwise, this extra 10% from the EU would be eaten up by only two years of emissions increases in China, and we will still not have saved the clima…
- Reference: Autrement, ces 10 % supplémentaires de la part de l'UE seront anéantis par seulement deux années d'augmentation des émissions en Chine et nous n'auro…
- Beam: Sinon, ces 10 % supplémentaires de l'UE seraient ennuyés par deux années seulement d'augmentations des émissions en Chine, et nous n'aurons toujours …
- Selected: Sinon, ce 10 % supplémentaire provenant de l'UE serait épuisé par deux ans seulement d'augmentations des émissions en Chine, et nous n'aurons toujour…

### Most hurt examples

**en-fr / en/ep-07-09-25-012.xml / 106** (COMET -0.6051)

- Previous: You are therefore considering whether you should say that at least 50% must in future go to these alternatives, because my home city of Berlin argues…
- Source: I think that is mad!
- Reference: Je trouve cela aberrant !
- Beam: Je pense que c'est folle !
- Selected: Je pense que c'est de l'aile !

**en-fr / en/ep-09-11-24-003.xml / 375** (COMET -0.1644)

- Previous: (IT) Mr President, ladies and gentlemen, let us draw a merciful veil over the unbelievable sum of EUR 30 billion per year until 2020 that the EU woul…
- Source: To us Italians, that is reminiscent of the notorious Italian Cassa del Mezzogiorno.
- Reference: Cela nous rappelle, à nous, les Italiens, la célèbre Cassa del Mezzogiorno italienne.
- Beam: Pour nous, les Italiens, cela rappelle la célèbre Cassa del Mezzogiorno italienne.
- Selected: Pour nous, Italiens, cela rappelle le fameux cassa del Mezzogiorno italien.

**en-de / en/ep-05-06-08.xml / 1039** (COMET -0.1601)

- Previous: & # 160; & # 160; Mr President, this concerns the way in which we interpret the Rules of Procedure.
- Source: I arranged with the Secretary-General to have a one-minute explanation.
- Reference: Ich habe mit dem Generalsekretär eine Redezeit von einer Minute vereinbart.
- Beam: Ich habe mit dem Generalsekretär eine einminütige Erklärung vereinbart.
- Selected: Ich habe mit dem Generalsekretär eine Wortmeldung von einer Minute getroffen.

**en-de / en/ep-09-02-03-012.xml / 255** (COMET -0.1593)

- Previous: We must establish quotas for refugees which are distributed across the different Member States and which also include the refugees who come across th…
- Source: These represent quite a significant number.
- Reference: Das sind nämlich auch nicht wenige.
- Beam: Das ist eine ziemlich große Zahl.
- Selected: Diese beziehen sich auf eine beträchtliche Anzahl.

**en-de / en/ep-96-12-13.xml / 1210** (COMET -0.1488)

- Previous: This report by the Court of Auditors is thus politically heavily loaded and Parliament cannot allow that to happen in future.
- Source: It cannot be allowed to get through the forthcoming Intergovernmental Conference on the nod.
- Reference: Dies kann an der kommenden Regierungskonferenz nicht unbemerkt vorübergehen.
- Beam: Es kann nicht gestattet werden, dass die bevorstehende Regierungskonferenz über den Anschlag hinweggeht.
- Selected: Es darf nicht zugelassen werden, dass es sich bei der nächsten Regierungskonferenz um den Beifall handelt.

## Interpretation

Compare meaning and omissions manually in the source, reference and two outputs. Report both corpus BLEU/chrF and COMET; a positive COMET delta on one sentence is an inspection cue, not a significance test.
