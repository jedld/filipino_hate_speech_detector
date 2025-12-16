# Model Comparison Report: Small Transformer vs Llama 3

## Summary Statistics
- Total Test Samples: 2271
- Exp 1 Accuracy: 0.6918
- Exp 3 Accuracy: 0.8692

## Disagreement Analysis
- **Exp 1 Correct, Exp 3 Wrong**: 135 samples
- **Exp 3 Correct, Exp 1 Wrong**: 538 samples
- **Both Correct**: 1436 samples
- **Both Wrong**: 162 samples

### Exp 1 Correct, Exp 3 Wrong (Small Model Wins)
**Text:** Di magtataka yung mga tao eh. The one of us mentality.

- Label: 1
- Exp 1 Pred: 1 (Conf: 0.92)
- Exp 3 Pred: 0 (Conf: 0.87)
---
**Text:** The Benghazi hearings. It's the American version of the never ending Binay hearings we have here.

- Label: 1
- Exp 1 Pred: 1 (Conf: 1.00)
- Exp 3 Pred: 0 (Conf: 0.74)
---
**Text:** What is the name of Binay's fandom? #BINAYaran. :)) Basta ako #JaDine lang.

- Label: 1
- Exp 1 Pred: 1 (Conf: 0.99)
- Exp 3 Pred: 0 (Conf: 0.99)
---
**Text:** I thought matino pa tong si APC. Now Im beginning to think he is behind black props vs Mar. Not excluding Binay yet. https://t.co/vh40H8h9Xj

- Label: 1
- Exp 1 Pred: 1 (Conf: 1.00)
- Exp 3 Pred: 0 (Conf: 0.76)
---
**Text:** #Halalan2016

- Label: 0
- Exp 1 Pred: 0 (Conf: 0.80)
- Exp 3 Pred: 1 (Conf: 0.65)
---
**Text:** Bakit ang payat at haggard ni grace poe? #inthenews

- Label: 0
- Exp 1 Pred: 0 (Conf: 0.99)
- Exp 3 Pred: 1 (Conf: 0.85)
---
**Text:** Binay didn't know that under Daang Matuwid, 7.7M Filipinos escaped poverty. Let me repeat: 7.7M are no longer poor. #politicsph

- Label: 1
- Exp 1 Pred: 1 (Conf: 0.96)
- Exp 3 Pred: 0 (Conf: 0.75)
---
**Text:** "Pero di tulad ng iba, maraming nagawa. Yan si Binay." AHAHAHAHAHAHAHAAHAHAHAHAHAHAHAAHAHAHAHAHAHAHAHAHAHAHAHAHAHAHAHAHA.

- Label: 1
- Exp 1 Pred: 1 (Conf: 0.86)
- Exp 3 Pred: 0 (Conf: 0.95)
---
**Text:** Hindi ako makaget over dun sa Hala Nahuloglog jokes. Lalo na yung Hala Nasu-nog-nog ni Binay ??

- Label: 0
- Exp 1 Pred: 0 (Conf: 0.99)
- Exp 3 Pred: 1 (Conf: 0.61)
---
**Text:** Fight For KAMZO

- Label: 0
- Exp 1 Pred: 0 (Conf: 0.94)
- Exp 3 Pred: 1 (Conf: 0.78)
---
**Text:** I always heard about Binay, Poe, Roxas candidacy campaign. But Duterte??? Damn Media!
#ALDUB32ndWeeksary

- Label: 0
- Exp 1 Pred: 0 (Conf: 0.99)
- Exp 3 Pred: 1 (Conf: 0.76)
---
**Text:** #DU30forPresident

- Label: 1
- Exp 1 Pred: 1 (Conf: 0.68)
- Exp 3 Pred: 0 (Conf: 0.79)
---
**Text:** I don't really see the point of having Binay as a President of this country. Nor Mar Roxas. Nor Grace Poe.

- Label: 1
- Exp 1 Pred: 1 (Conf: 0.79)
- Exp 3 Pred: 0 (Conf: 0.95)
---
**Text:** https://t.co/4erihjEZwM

- Label: 1
- Exp 1 Pred: 1 (Conf: 0.72)
- Exp 3 Pred: 0 (Conf: 0.64)
---
**Text:** DUTERTE-CAYETANO 4EVER.  https://t.co/iMzb55G1qS

- Label: 1
- Exp 1 Pred: 1 (Conf: 1.00)
- Exp 3 Pred: 0 (Conf: 0.95)
---
**Text:** @gmanews so dahil Kay Binay na naman to? hahaha

- Label: 1
- Exp 1 Pred: 1 (Conf: 0.81)
- Exp 3 Pred: 0 (Conf: 1.00)
---
**Text:** "@Team_Inquirer: Binay not speaking for gov’t—Palace http://t.co/gftOJjvksU | @TJBurgonio" bwahahaha!

- Label: 1
- Exp 1 Pred: 1 (Conf: 0.99)
- Exp 3 Pred: 0 (Conf: 0.98)
---
**Text:** Sino-sino ang nasa line-up ng UNA na respectable daw?Unang-una , kung si Binay ang kakampihan mong pres, respectable ba yun? #SubtweetinMe

- Label: 1
- Exp 1 Pred: 1 (Conf: 0.99)
- Exp 3 Pred: 0 (Conf: 0.62)
---
**Text:** Check link: VP#JejomarBinay LIAR, lying about his #EducationalAttainmentshttps://t.co/0MCK9qgEh2#Binay #Binay2016 https://t.co/lcJ1GDB3mi

- Label: 0
- Exp 1 Pred: 0 (Conf: 0.99)
- Exp 3 Pred: 1 (Conf: 0.85)
---
**Text:** @FashionPulis wow, supporting Mar Roxas?

- Label: 0
- Exp 1 Pred: 0 (Conf: 0.98)
- Exp 3 Pred: 1 (Conf: 0.68)
---

### Exp 3 Correct, Exp 1 Wrong (Llama 3 Wins)
**Text:** Para sa labanan sa pagkaalkalde sa Makati sa darating eleksyon, sa ilalim ng UNA, iprinoklama si Rep. Abby Binay bilang kanilang pambato.

- Label: 0
- Exp 1 Pred: 1 (Conf: 1.00)
- Exp 3 Pred: 0 (Conf: 1.00)
---
**Text:** Poe advisers say they’re more than a match for Binay’s Puno https://t.co/Kc8J4lCuqa https://t.co/Ym3KmMH6zr

- Label: 0
- Exp 1 Pred: 1 (Conf: 0.99)
- Exp 3 Pred: 0 (Conf: 1.00)
---
**Text:** Napaka-PUTANGINA niyo po Mar Roxas. Swear!

- Label: 1
- Exp 1 Pred: 0 (Conf: 0.52)
- Exp 3 Pred: 1 (Conf: 1.00)
---
**Text:** THE STANDARD: UPDATE: Sen. Miriam Santiago’s ‘meet &amp; greet’ event at Bahay ng Alumni in UP Diliman begins at 6 pm Monday.

- Label: 0
- Exp 1 Pred: 1 (Conf: 0.99)
- Exp 3 Pred: 0 (Conf: 1.00)
---
**Text:** DI GUMAGANA UNG MIC NI BINAY HAHAHAHAHAHA #PiliPinasDebates2016

- Label: 0
- Exp 1 Pred: 1 (Conf: 0.98)
- Exp 3 Pred: 0 (Conf: 0.94)
---
**Text:** #shareLang... https://t.co/Yu3bccCdnl

- Label: 0
- Exp 1 Pred: 1 (Conf: 0.99)
- Exp 3 Pred: 0 (Conf: 0.89)
---
**Text:** Mar Roxas is this and that. Isnt about time you compare him to other candidates? https://t.co/OjeSdTfuCm

- Label: 0
- Exp 1 Pred: 1 (Conf: 0.98)
- Exp 3 Pred: 0 (Conf: 0.99)
---
**Text:** Mar + Poe against Duterte? Nako SWAYANG POE. #DuterteTillTheEnd #DuterteForPresident #Duterte2016

- Label: 0
- Exp 1 Pred: 1 (Conf: 0.96)
- Exp 3 Pred: 0 (Conf: 0.79)
---
**Text:** Ahhh so people would still choose Mar and Binay over MDS...but why? ?? https://t.co/DPZAWofynX

- Label: 0
- Exp 1 Pred: 1 (Conf: 0.84)
- Exp 3 Pred: 0 (Conf: 1.00)
---
**Text:** We still support Mar Roxas. You can't bring a good man down. :)

- Label: 0
- Exp 1 Pred: 1 (Conf: 0.87)
- Exp 3 Pred: 0 (Conf: 0.99)
---
**Text:** Presedential bet Grace Poe immediately REJECTED administration standard-bearer Mar Roxas's call for unity... https://t.co/1Orltfel00

- Label: 0
- Exp 1 Pred: 1 (Conf: 0.93)
- Exp 3 Pred: 0 (Conf: 1.00)
---
**Text:** 'Di naman ako si Binay. Pero bakit hindi mo makitang andito lang ako para sa'yo. #HugotHalalan https://t.co/KmRrMwSYbZ

- Label: 0
- Exp 1 Pred: 1 (Conf: 0.93)
- Exp 3 Pred: 0 (Conf: 0.97)
---
**Text:** RT @SagadaSun: These elites will RULE no matter who wields political power. But I think they are afraid of Binay and egged his demolition. …

- Label: 0
- Exp 1 Pred: 1 (Conf: 0.79)
- Exp 3 Pred: 0 (Conf: 0.94)
---
**Text:** @deejayap @mgvtrb Ok. So Bautista's FOR Poe, Guanzon's FOR Roxas. Who's FOR Binay, Duterte and the rest? Commission FOR (not ON) Elections.

- Label: 0
- Exp 1 Pred: 1 (Conf: 0.99)
- Exp 3 Pred: 0 (Conf: 0.99)
---
**Text:** Natawa ako sa news abt BINAY!https://t.co/tTsbPrmMRn nya w/"kana"chix 2.Q kng pano nya sinulot ang VP position?3.P-NAY na raw if ever

- Label: 0
- Exp 1 Pred: 1 (Conf: 1.00)
- Exp 3 Pred: 0 (Conf: 0.99)
---
**Text:** BACOLOD MAYORALTY AS OF 7:17PM

1. DUTERTE, RODY (PDPLBN); 7,034,664
2. POE, GRACE (IND;) 4,012,417
3. ROXAS, MAR... https://t.co/P79fSZ0qej

- Label: 0
- Exp 1 Pred: 1 (Conf: 1.00)
- Exp 3 Pred: 0 (Conf: 1.00)
---
**Text:** Report: Mayor Duterte says presidential run ‘now on the table’.
VP Binay: My presidential run is "under the table". https://t.co/ktDLIInutI

- Label: 0
- Exp 1 Pred: 1 (Conf: 1.00)
- Exp 3 Pred: 0 (Conf: 1.00)
---
**Text:** CarGlen Wala Nang Iba

- Label: 0
- Exp 1 Pred: 1 (Conf: 0.95)
- Exp 3 Pred: 0 (Conf: 0.54)
---
**Text:** Wag ng umasa Mar Roxas masakit umasa.

- Label: 1
- Exp 1 Pred: 0 (Conf: 0.98)
- Exp 3 Pred: 1 (Conf: 0.93)
---
**Text:** @jepoi_ordaniel He has enough time. A lot can still happen to Binay, Santiago and Llamanzares. ??

- Label: 0
- Exp 1 Pred: 1 (Conf: 0.99)
- Exp 3 Pred: 0 (Conf: 0.99)
---

### Both Wrong (Hard Samples)
**Text:** @theJonnny malakas @sengracepoe &amp; alam ni Binay yn so less votes ky @MARoxas na mortal enemy nia, @SayChiz looks like loyal ally @twitnigab

- Label: 0
- Exp 1 Pred: 1 (Conf: 0.98)
- Exp 3 Pred: 1 (Conf: 0.69)
---
**Text:** @Antifornicator yes the govenment's secret service sprinkled magic vetsin to vote binay or mar...halaka u will vote for either of the 2! lol

- Label: 0
- Exp 1 Pred: 1 (Conf: 1.00)
- Exp 3 Pred: 1 (Conf: 0.92)
---
**Text:** Kayo talaga naaalala ko sa commercial ni binay ./. Kayo ni Dawming Jyung-jyung ?????? @MaricarPD_

- Label: 0
- Exp 1 Pred: 1 (Conf: 1.00)
- Exp 3 Pred: 1 (Conf: 0.98)
---
**Text:** #LabanLeni naalala ko sabi ng mga taga ARMM...si bbm ang may plan b-z! Baba daw si duterte.impeach daw si binay.

- Label: 0
- Exp 1 Pred: 1 (Conf: 1.00)
- Exp 3 Pred: 1 (Conf: 0.92)
---
**Text:** Has Grace Poe criticized Jalosjos for actually raping a child? NO.

- Label: 0
- Exp 1 Pred: 1 (Conf: 0.81)
- Exp 3 Pred: 1 (Conf: 0.69)
---
**Text:** My Uber driver asked me kung sino ang presidente ko. I just answered "kahit sino, 'wag lang si Binay" ???? #Halalan2016

- Label: 0
- Exp 1 Pred: 1 (Conf: 0.99)
- Exp 3 Pred: 1 (Conf: 0.57)
---
**Text:** JOSHANELT FOR BLOODYCRAYONS https://t.co/F6DtlJ6qUJ

- Label: 0
- Exp 1 Pred: 1 (Conf: 0.99)
- Exp 3 Pred: 1 (Conf: 0.63)
---
**Text:** Not a Poe supporter nor hater, but good thing Supreme Court got its senses. We cannot have Binay as our president, please.

- Label: 0
- Exp 1 Pred: 1 (Conf: 0.69)
- Exp 3 Pred: 1 (Conf: 0.87)
---
**Text:** PH Vote Duterte ????????

- Label: 1
- Exp 1 Pred: 0 (Conf: 0.73)
- Exp 3 Pred: 0 (Conf: 0.60)
---
**Text:** RT @Dyan_Kanalang: No money pero 2nd in spending at may pang Spotify pa POE. Paki paliwanag nga, Chiz. O busy ka sa Plan Binay mo? https://…

- Label: 0
- Exp 1 Pred: 1 (Conf: 0.99)
- Exp 3 Pred: 1 (Conf: 0.77)
---
**Text:** wag nyo nman ibash yung anak ni binay

- Label: 1
- Exp 1 Pred: 0 (Conf: 0.87)
- Exp 3 Pred: 0 (Conf: 0.58)
---
**Text:** RT @_CEREALKILL3R: President Putin of Russia. No entourage. VS our VP and BSP President. Tangna talaga si Binay. Pabebe? https://t.co/Kj4nL…

- Label: 0
- Exp 1 Pred: 1 (Conf: 0.63)
- Exp 3 Pred: 1 (Conf: 0.97)
---
**Text:** Nognog = Kai

- Label: 0
- Exp 1 Pred: 1 (Conf: 0.99)
- Exp 3 Pred: 1 (Conf: 0.96)
---
**Text:** Hindi po ako naniniwalang magkalapit si Roxas at Duterts. This elections is Duterte vs Poe. Srsly.

- Label: 0
- Exp 1 Pred: 1 (Conf: 0.97)
- Exp 3 Pred: 1 (Conf: 0.93)
---
**Text:** Yan Si binay! haha pat.  https://t.co/Av2STrH8Cu

- Label: 0
- Exp 1 Pred: 1 (Conf: 0.99)
- Exp 3 Pred: 1 (Conf: 0.55)
---
**Text:** Mar Roxas, pabor na 'magpagulong' ng mga ulo sa airport ; Sen. Escudero, binanatan si Abaya sa 'tanim-bala' #BreaktimeHeadlines

- Label: 0
- Exp 1 Pred: 1 (Conf: 1.00)
- Exp 3 Pred: 1 (Conf: 0.53)
---
**Text:** Kung alam ko lang na tatakbo si Duterte, hindi ko na sana ako sa Fruitas na may mukha ni Binay.

- Label: 0
- Exp 1 Pred: 1 (Conf: 0.99)
- Exp 3 Pred: 1 (Conf: 0.72)
---
**Text:** Sa  UPLB pa pumunta si VP Binay. Aktibista mga students dun ??

- Label: 1
- Exp 1 Pred: 0 (Conf: 0.52)
- Exp 3 Pred: 0 (Conf: 0.99)
---
**Text:** Pati bala tinatanim na... edible na din ang bala cgro ano PNOY? Ipagmalaki mopa sya Mar Roxas at ipagpapatuloy mo... https://t.co/9ueKrq1BdT

- Label: 1
- Exp 1 Pred: 0 (Conf: 0.99)
- Exp 3 Pred: 0 (Conf: 1.00)
---
**Text:** Lahat ng advertisement ni binay sa pangangandidato eh puro self pity???

- Label: 0
- Exp 1 Pred: 1 (Conf: 0.99)
- Exp 3 Pred: 1 (Conf: 0.99)
---

