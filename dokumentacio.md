# Intelligens Épületenergetikai Termosztát Modell Prediktív Irányítása (MPC)

Ez a projekt egy **Modell Prediktív Irányítás (Model Predictive Control - MPC)** alapú diszkrét idejű szabályozási rendszert valósít meg egy lakóépület belső hőmérsékletének optimalizálására. A projekt a mintavételezett állapottér-modellezés, a konvex optimalizálás és a gördülő horizontú (receding horizon) vezérlés elveire épül.

A hagyományos, reaktív szabályozókkal (mint az On-Off termosztátok vagy a PID szabályozók) ellentétben ez az MPC architektúra **proaktív módon** avatkozik be: a predikciós horizontján belül előre látja a jövőbeli referenciaértékeket és környezeti zavarásokat, így képes azokat még a káros tranziensek kialakulása előtt kompenzálni.

---

## 1. Elméleti Háttér és Architektúra

Az MPC lényege, hogy a szabályozott szakasz (szoba) matematikai modelljét felhasználva a jelenlegi időpillanatban ($t$) kiszámítja a jövőbeli beavatkozójelek egy optimális sorozatát egy véges horizonton ($N$) belül. Ebből a sorozatból csak a legelső elemet valósítja meg, majd a következő mintavételi pillanatban a horizontot eltolva (gördülő horizont elv) újraoptimalizálja a rendszert a friss állapot-visszacsatolás alapján.

### Miért jobb az MPC ebben a feladatban, mint egy PID?

1. **Kényszerek Kezelése (Constraint Handling):** A valós fűtési rendszerek korlátosak (nem tudunk negatív irányba fűteni, azaz hűteni klíma nélkül, és a kazán maximális teljesítménye is véges). A PID nem tudja expliciten kezelni ezeket a korlátokat (gyakran fellép az _integrátor-telítődés / wind-up_ jelenség). Az MPC az optimalizálási feladat kényszereként közvetlenül integrálja ezeket a fizikai határokat.
2. **Anticipációs Képesség (Holtidő-kompenzáció):** Mivel az épületek nagy termikus tehetetlenséggel és holtidővel rendelkeznek, egy PID csak akkor kezd el fűteni, ha a hőmérséklet már visszaesett. Az MPC ezzel szemben "látja a jövőt", így képes hamarabb elkezdeni a fűtést (előfűtés).
3. **Zavarások Előretekintése (Disturbance Feed-forward):** Az időjárás-előrejelzésekből vagy napi rutinokból származó determinisztikus zavarások (napsütés, külső lehűlés) beépíthetők a predikciós modellbe, így a rendszer nem utólag reagál a zavarásra, hanem felkészül rá.

---

## 2. A Rendszer Matematikai Modellje

A szoba termikus viselkedését egy elsőrendű lineáris differenciálegyenlettel modellezzük, amelyet Euler-féle diszkretizációval ($dt = 1.0$ perc mintavételi idővel) az alábbi diszkrét idejű differenciaegyenlet formájára hoztunk:

$$T_{k+1} = T_k + dt \cdot \Big[ -a \cdot (T_k - T_{out,k}) + b \cdot u_k + Q_{dist,k} \Big]$$

### A modell komponenseinek fizikai magyarázata:

- **$T_k$ [°C]:** A belső szobahőmérséklet (állapotváltozó) a $k$-adik időlépésben.
- **$T_{out,k}$ [°C]:** A külső környezeti hőmérséklet (időben változó környezeti zavarás).
- **$u_k$ [%]:** A fűtőberendezés vezérlőjele (beavatkozó jel), értéke $0$ és $100$ % között változhat.
- **$Q_{dist,k}$ [°C/perc]:** Extra belső hőterhelés (pl. elektromos berendezések disszipációja vagy az ablakon besütő nap sugárzási energiája).
- **$a = 0.01$ [1/perc]:** Hőveszteségi tényező. Kifejezi, hogy a falak hőszigetelése milyen gyorsan engedi kiegyenlítődni a belső és külső hőmérsékletet. A kis érték kiváló hőszigetelésű, nagy tehetetlenségű épületet reprezentál.
- **$b = 0.04$ [°C / (% $\cdot$ perc)]:** Fűtési hatékonysági tényező, amely megadja, hogy $1\%$ fűtési teljesítmény egy perc alatt hány fokot képes emelni a szoba hőmérsékletén.

---

## 3. Az Optimalizálási Probléma és Költségfüggvény

Minden időlépésben egy kvadratikus programozási (QP) feladatot oldunk meg a `cvxpy` segítségével. A predikciós horizont hossza $N = 30$ időlépés (perc).

### 3.1. A Célfüggvény (Cost Function) Matematikai Formulája

A szabályozási célok eléréséhez a következő kvadratikus költségfüggvényt minimalizáljuk a horizont mentén:

$$\min_{u} J = \sum_{k=0}^{N-1} q \cdot (T_k - T_{ref,k})^2 + \sum_{k=1}^{N-1} r \cdot (u_k - u_{k-1})^2$$

Ahol a súlyozó tényezők beállítása:

- **$q = 1.0$ (Hőmérséklet követési súly):** A referenciaértéktől való eltérés négyzetes büntetése. Ez kényszeríti ki a szobahőmérséklet pontos beállását.
- **$r = 0.5$ (Beavatkozójel változási sebesség súly / Slew Rate Penalty):** Ez a tag nem magát a fűtési energiát ($u$) bünteti, hanem annak időlépések közötti eltérését ($\Delta u = u_k - u_{k-1}$).

Ha közvetlenül az $u^2$ energiát büntetnénk, a vezérlő a szűkös fűtési hatékonyság ($b$) miatt inkább hagyná kihűlni a lakást, mert a fűtés fenntartása túl "drága" lenne a solver számára. A $\Delta u$ büntetés alkalmazásával a fűtés elérheti a 100%-ot, de garantálja, hogy a vezérlő simán, ugrásoktól és oszcillációktól (chattering) mentesen működtesse a szelepeket, megvédve a fizikai aktuátorokat a gyors elhasználódástól.

### 3.2. Fizikai Korlátozások (Constraints)

A fizikai megvalósíthatóság érdekében az optimalizálónak szigorú egyenlőtlenségi kényszereket kell betartania:
$$0 \le u_k \le 100 \quad \forall k \in [0, N-1]$$

Mivel a rendszer aszimmetrikus (csak fűteni tud, aktív hűtőegység/klíma nem áll rendelkezésre), az $u_k \ge 0$ alsó korlát kulcsfontosságú. Ha a szoba a külső zavarás miatt túlmelegszik, a szabályozó nem képes negatív fűtést alkalmazni, csak a fűtés lekapcsolásával ($u=0$) megvárni, amíg a rendszer a falakon át vissza nem hűl.

---

## 4. Dinamikus Zavarások és Környezet Modellezése

A szimuláció egy teljes 24 órás (1440 perc) periódust fed le, valósághű és dinamikusan változó környezeti profilokkal:

1. **Külső hőmérséklet-ingadozás ($T_{out}$):** Nem egy fix érték, hanem egy 24 órás periódusidejű szinuszfüggvény modellezi a nappali felmelegedést és az éjszakai lehűlést:
   $$T_{out}(t) = 7.0 + 5.0 \cdot \sin\left(\frac{2\pi \cdot t}{1440} - \frac{\pi}{2}\right)$$
   Ez éjfél utáni minimumot ($2^\circ\text{C}$) és délutáni maximumot ($12^\circ\text{C}$) eredményez.
2. **Változó Referencia ütemterv ($T_{ref}$):** A lakók igényeihez igazodó termosztát-profil:
   - **Éjszakai / Energiatakarékos fázis (22:00 - 08:00):** $18.0^\circ\text{C}$
   - **Nappali / Komfort fázis (08:00 - 22:00):** $22.0^\circ\text{C}$
3. **Belső hőterhelés ($Q_{dist}$):** A 600. és 900. perc között (10:00 és 15:00 óra között) egy konstans $0.06^\circ\text{C}/\text{perc}$ értékű extra hőbevitel jelenik meg a rendszerben, szimulálva az ablakon besütő napot vagy az épületben tartózkodó emberek által termelt hőt.

---

## 5. Szimulációs Eredmények Elemzése

A lefutott rendszer szimulációs görbéi tökéletesen szemléltetik az MPC három legfontosabb rendszermérnöki tulajdonságát:

- **Prediktív Anticipáció (Előfűtés):** A 480. percben (reggel 08:00) a referenciaérték hirtelen felugrik $18^\circ\text{C}$-ról $22^\circ\text{C}$-ra. Az MPC a horizontján ($N=30$) ezt előre látja, így már a 450. perctől kezdve fokozatosan elkezdi feltekerni a fűtést ($u$). Ennek köszönhetően a szobahőmérséklet egy szép, lágy exponenciális görbe mentén pontosan a váltás pillanatára eléri a kívánt komfortfokozatot, túllövés nélkül.
- **Intelligens Zavarás-elhárítás (Disturbance Rejection):** A 600. és 900. perc között belépő belső hőterhelés hatására a belső hőmérséklet hajlamos lenne túlmelegedni. Az MPC azonban érzékeli ezt a plusz energiát, és azonnal, a zavarással szinkronban **visszaszabályozza a fűtési teljesítményt**. A fűtés lecsökkenése közvetlen energiamegtakarítást jelent, miközben a belső hőmérséklet-vonal teljesen lapos és zavarmentes marad.
- **Aktuátor Védelem:** A $\Delta u$ büntetés miatt a fűtési teljesítmény görbéje teljesen folytonos, nincsenek benne nagy, hirtelen ugrások vagy nagyfrekvenciás kapcsolgatások, ami a valódi fizikai megvalósíthatóság alapfeltétele.

---

## 6. Telepítési és Futtatási Útmutató

A szimulációs környezet futtatásához Python 3 és néhány alapvető matematikai/optimalizációs könyvtár szükséges. A szoftver hordozható, nem igényel bonyolult telepítést vagy dedikált szerverkörnyezetet.

### 6.1. Virtuális környezet (Virtual Environment) beállítása – _Ajánlott_

A függőségek telepítése előtt erősen ajánlott egy izolált virtuális környezet (`venv`) létrehozása. Ennek fő előnye, hogy a projekt megkapja a saját, elkülönített csomagjait és verzióit. Így elkerülhetők a verziókonfliktusok a számítógépen lévő más Python projektekkel, és a rendszer globális környezete is tiszta marad.

Nyiss meg egy terminált (vagy parancssort) a projekt mappájában, és futtasd az alábbi parancsokat:

```bash
# 1. Virtuális környezet létrehozása (a "venv" mappába)
python -m venv venv

# 2. A virtuális környezet aktiválása:
# Windows rendszeren:
venv\Scripts\activate

# macOS / Linux rendszeren:
source venv/bin/activate
```

_(Sikeres aktiválás esetén a terminál parancssora előtt meg kell jelennie a `(venv)` feliratnak.)_

### 6.2. Szükséges könyvtárak telepítése

A projekt a `cvxpy` (konvex optimalizáció), a `numpy` (numerikus számítások) és a `matplotlib` (adatvizualizáció) csomagokra épül. Mivel egy notebook fájlt fogunk futtatni, szükség lesz a `notebook` csomagra is.

Az aktivált virtuális környezetben futtasd az alábbi parancsot:

```bash
pip install numpy cvxpy matplotlib notebook
```

### 6.3. A szimuláció végrehajtása lépésről lépésre

1. **A környezet megnyitása:** Ugyanebben az aktivált terminálban indítsd el a Jupyter környezetet a következő paranccsal:
   ```bash
   jupyter notebook
   ```
   Ekkor megnyílik a böngésződ. Tallózd be és nyisd meg a `temperature_controller_daynight.ipynb` fájlt.
   _(Alternatíva: A fájl telepítés és virtuális környezet nélkül is futtatható felhős környezetben, például a Google Colab felületén.)_
2. **A kód végrehajtása:** Futtasd le a notebook összes celláját fentről lefelé haladva (a menüben: **Kernel -> Restart & Run All**).
3. **Számítási idő és visszajelzés:** Mivel a szimuláció egy teljes 24 órás (1440 perces) időtartamot ölel fel, és percenként megold egy 30 lépéses horizontra vonatkozó QP (Kvadratikus Programozási) feladatot, a futás a gép teljesítményétől függően **10-30 másodpercet** vesz igénybe. A kimeneten folyamatosan frissülő státuszüzenetek jelzik a számítás előrehaladását.
4. **Eredmények megtekintése:** A számítás lefutását követően a kód automatikusan generálja és megjeleníti a kétsávos, részletes grafikont:
   - **Felső ábra (Hőmérséklet-követés):** A szobahőmérséklet, a referencia és a külső hőmérséklet alakulása. Ezen a görbén jól megfigyelhető az MPC proaktív előfűtési stratégiája.
   - **Alsó ábra (Beavatkozó jel és Zavarás):** A fűtési teljesítmény és a belső hőterhelés kompenzációjának alakulása, szemléltetve az algoritmus energiatakarékos visszaszabályozását a napsütéses időszakban.

---

## 7. Továbbfejlesztési Lehetőségek (Future Work)

A jelenlegi architektúra robusztus és stabil, de kiváló alapot nyújt bonyolultabb épületautomatizálási rendszerek modellezéséhez is. Egyetemi kutatás (pl. TDK vagy Szakdolgozat) vagy ipari kiterjesztés céljából az alábbi irányokba fejleszthető tovább:

- **Többzónás (MIMO) Irányítás:** Valós épületeknél a szobák (zónák) nincsenek elszigetelve egymástól. A rendszer kiterjeszthető többváltozós formára, ahol az egymás melletti szobák közötti hőcsere (a falakon és ajtókon keresztül) csatolt "kereszthatásként" jelenik meg az állapottér-mátrixokban.
- **Gazdasági MPC (Economic MPC):** A pontos referenciaértékek merev követése helyett egy megengedett "komfort-tartomány" (pl. 21 °C és 23.5 °C között) meghatározása. A költségfüggvény ebben az esetben nem a követési hibát, hanem a valós idejű, dinamikus energiaárak (csúcsidőszaki vs. éjszakai áram/gáz tarifa) alapján a tényleges pénzügyi költséget minimalizálja.
- **Állapot- és Zavarásbecslés (Kalman-szűrő):** A valóságban a falak pontos hőátbocsátása és a napsugárzás mértéke nem ismert tökéletesen, a szenzorok pedig zajosak. Egy Kalman-szűrő (vagy kiterjesztett állapotfigyelő) beépítésével a rendszer képessé válna a mérési zajok szűrésére és az ismeretlen külső zavarások robusztus, online becslésére.

---

**Szerző:** Gáspár Tamás  
**Intézmény:** Sapientia Erdélyi Magyar Tudományegyetem  
**Tantárgy:** Modell Prediktív Irányítások
