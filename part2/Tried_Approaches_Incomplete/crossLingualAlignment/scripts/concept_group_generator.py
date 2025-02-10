# Sample input string (you would replace this with your full data)
data = """
numbers एक one numbers दो two numbers तीन three numbers चार four numbers पांच five
numbers छह six numbers सात seven numbers आठ eight numbers नौ nine numbers दस ten
time सुबह morning time दोपहर afternoon time शाम evening time रात night time कल tomorrow
time आज today time परसों day-after-tomorrow time सप्ताह week time महीना month time साल year
colors लाल red colors नीला blue colors हरा green colors पीला yellow colors काला black
colors सफेद white colors बैंगनी purple colors गुलाबी pink colors भूरा brown colors नारंगी orange
family माता mother family पिता father family बहन sister family भाई brother family दादी grandmother
family दादा grandfather family चाचा uncle family चाची aunt family बेटा son family बेटी daughter
food रोटी bread food दाल lentils food चावल rice food सब्जी vegetables food फल fruits
food दूध milk food पानी water food चाय tea food कॉफी coffee food मसाला spices
body सिर head body आंख eye body नाक nose body कान ear body मुंह mouth body हाथ hand body पैर foot
body पेट stomach body दिल heart body दांत teeth
animals कुत्ता dog animals बिल्ली cat animals गाय cow animals घोड़ा horse animals चूहा rat
animals हाथी elephant animals शेर lion animals बंदर monkey animals चिड़िया bird animals मछली fish
nature पेड़ tree nature फूल flower nature पत्ता leaf nature बादल cloud nature सूरज sun
nature चांद moon nature तारे stars nature नदी river nature पहाड़ mountain nature समुद्र ocean
clothes कमीज shirt clothes पैंट pants clothes साड़ी saree clothes कुर्ता kurta clothes जूते shoes
clothes मोजे socks clothes टोपी hat clothes दुपट्टा scarf clothes जैकेट jacket clothes कोट coat
emotions खुशी happiness emotions दुख sadness emotions प्यार love emotions गुस्सा anger emotions डर fear
emotions आशा hope emotions चिंता worry emotions उत्साह excitement emotions थकान tiredness emotions तनाव stress
weather गरमी heat weather सर्दी cold weather बारिश rain weather धूप sunshine weather आंधी storm
weather कोहरा fog weather ओस dew weather बर्फ snow weather हवा wind weather तूफान hurricane
directions उत्तर north directions दक्षिण south directions पूर्व east directions पश्चिम west directions ऊपर up
directions नीचे down directions अंदर inside directions बाहर outside directions आगे forward directions पीछे backward
professions डॉक्टर doctor professions शिक्षक teacher professions वकील lawyer professions इंजीनियर engineer professions पायलट pilot
professions नर्स nurse professions किसान farmer professions दर्जी tailor professions बढ़ई carpenter professions पुलिस police
vehicles कार car vehicles बस bus vehicles ट्रेन train vehicles विमान airplane vehicles नाव boat
vehicles साइकिल bicycle vehicles स्कूटर scooter vehicles ट्रक truck vehicles मोटरसाइकिल motorcycle vehicles हेलीकॉप्टर helicopter
furniture मेज table furniture कुर्सी chair furniture पलंग bed furniture अलमारी cupboard furniture सोफा sofa
furniture शेल्फ shelf furniture दराज drawer furniture दर्पण mirror furniture पर्दा curtain furniture रैक rack
electronics टीवी television electronics फोन phone electronics कंप्यूटर computer electronics लैपटॉप laptop electronics प्रिंटर printer
electronics माइक्रोवेव microwave electronics फ्रिज refrigerator electronics एसी air-conditioner electronics रेडियो radio electronics कैमरा camera
fruits सेब apple fruits केला banana fruits संतरा orange fruits अंगूर grapes fruits आम mango
fruits पपीता papaya fruits अनार pomegranate fruits नाशपाती pear fruits कीवी kiwi fruits तरबूज watermelon
vegetables आलू potato vegetables टमाटर tomato vegetables प्याज onion vegetables गाजर carrot vegetables मटर peas
vegetables पालक spinach vegetables फूलगोभी cauliflower vegetables बैंगनी eggplant vegetables खीरा cucumber vegetables मूली radish
metals सोना gold metals चांदी silver metals लोहा iron metals तांबा copper metals पीतल brass
metals एल्युमिनियम aluminum metals प्लैटिनम platinum metals जस्ता zinc metals निकल nickel metals टिन tin
sports क्रिकेट cricket sports फुटबॉल football sports टेनिस tennis sports हॉकी hockey sports बैडमिंटन badminton
sports कबड्डी kabaddi sports वॉलीबॉल volleyball sports बास्केटबॉल basketball sports तैराकी swimming sports कुश्ती wrestling
instruments गिटार guitar instruments पियानो piano instruments बांसुरी flute instruments तबला tabla instruments वायलिन violin
instruments ड्रम drums instruments सितार sitar instruments हारमोनियम harmonium instruments बीन been instruments शहनाई shehnai
grains गेहूं wheat grains चावल rice grains मक्का corn grains जौ barley grains बाजरा millet
grains रागी ragi grains दाल lentils grains मटर peas grains राजमा kidney-beans grains छोले chickpeas
shapes वर्ग square shapes वृत्त circle shapes त्रिभुज triangle shapes आयत rectangle shapes गोला sphere
shapes षटभुज hexagon shapes पंचभुज pentagon shapes अष्टभुज octagon shapes डायमंड diamond shapes अंडाकार oval
months जनवरी january months फरवरी february months मार्च march months अप्रैल april months मई may
months जून june months जुलाई july months अगस्त august months सितंबर september months अक्टूबर october
buildings मकान house buildings विद्यालय school buildings अस्पताल hospital buildings दुकान shop buildings कार्यालय office
buildings होटल hotel buildings सिनेमा cinema buildings पुस्तकालय library buildings संग्रहालय museum buildings मंदिर temple
tools हथौड़ा hammer tools पेचकस screwdriver tools आरी saw tools चाकू knife tools कांटा fork
stationery पेन pen stationery पेंसिल pencil stationery रबड़ eraser stationery किताब book stationery कॉपी notebook
containers बोतल bottle containers डिब्बा box containers थैला bag containers बाल्टी bucket containers जार jar
seasons वसंत spring seasons ग्रीष्म summer seasons वर्षा monsoon seasons शरद autumn seasons हेमंत pre-winter
insects मक्खी fly insects मच्छर mosquito insects तितली butterfly insects भँवरा bee insects चींटी ant
flowers गुलाब rose flowers कमल lotus flowers सूरजमुखी sunflower flowers चमेली jasmine flowers गेंदा marigold
"""

# Split the entire data string into tokens (assuming tokens are separated by whitespace)
tokens = data.split()

# Create records by grouping every three tokens (category, Hindi, English)
records = []
for i in range(0, len(tokens), 3):
    # Check if there are at least 3 tokens left (to avoid incomplete records)
    if i + 2 < len(tokens):
        category = tokens[i].strip()
        hindi = tokens[i+1].strip()
        english = tokens[i+2].strip()
        records.append(f"{category} {hindi} {english}")

# For demonstration, let's print only a few sample records from selected categories:
sample_records = [rec for rec in records if rec.startswith("numbers") or rec.startswith("colors") or rec.startswith("family")]

for rec in sample_records:
    print(rec)

# write the records to a file
with open("data/concept_groups.txt", "w") as f:
    for rec in records:
        f.write(rec + "\n")