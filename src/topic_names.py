"""
HU: A 2/b altémák kézzel adott angol nevei (CFG.topic_version = "full2").
    Kulcs: (aspektus, témaszám) → (név, automatikus c-TF-IDF címke). Az automatikus címke
    „ujjlenyomat”: ha egy újrafuttatás más fürtöket ad (más témaszám = más tartalom), a
    get_topic_names() hibát dob, így a nevek nem kerülhetnek rossz témára.
    Szabályok: (1) a közel-duplikátum fürtök UGYANAZT a nevet kapják (az ábrák és a térképek
    a név szerint összevonnak); (2) a több dolgot felsoroló dicséret-fürtök „(list)” utótagot
    kapnak; (3) a kézi ellenőrzésen gyengébb (pontosság < 0,7) témák óvatos, tágabb nevet kapnak.
EN: Manually assigned English names of the 2/b subtopics (CFG.topic_version = "full2").
    Key: (aspect, topic id) → (name, automatic c-TF-IDF label). The automatic label is a
    "fingerprint": if a rerun yields different clusters (same id = different content),
    get_topic_names() raises, so names can never land on the wrong topic.
    Rules: (1) near-duplicate clusters get the SAME name (figures and maps merge by name);
    (2) praise clusters listing several things get a "(list)" suffix; (3) topics that were
    weaker in the manual check (precision < 0.7) get cautious, broader names.
"""
from __future__ import annotations

import pandas as pd

TOPIC_VERSION = "full2"


# HU/EN: panaszok / complaints
NEGATIVE_TOPIC_NAMES: dict[tuple[str, int], tuple[str, str]] = {
    # --- room ---
    ('room', 0): ('small, dated room', 'small / room / rooms'),
    ('room', 1): ('temperature & ventilation', 'air / room / hot'),
    ('room', 2): ('smell & cleanliness', 'smell / room / clean'),
    ('room', 3): ('poor lighting / curtains', 'light / lighting / dark'),
    ('room', 4): ('door & lock problems', 'door / lock / doors'),
    ('room', 5): ('no fridge / coffee in room', 'breakfast / coffee / tea'),
    ('room', 6): ('no space for luggage', 'luggage / stairs / space'),
    ('room', 7): ('small bathroom', 'bathroom / shower / small'),
    ('room', 8): ('dirty carpet', 'carpet / carpets / dirty'),
    ('room', 9): ('room price & charges', 'price / room / paid'),
    # --- location ---
    ('location', 0): ('hard to find or reach', 'not / hard / difficult'),
    ('location', 1): ('transport & taxi links', 'taxi / airport / bus'),
    ('location', 2): ('few restaurants & shops nearby', 'restaurants / restaurant / food'),
    ('location', 3): ('busy or steep road', 'road / property / busy'),
    ('location', 4): ('far from the centre', 'far / city / center'),
    ('location', 5): ('noisy surroundings', 'noisy / noise / loud'),
    ('location', 6): ('beach-related issues', 'beach / sea / not'),
    ('location', 7): ('area & neighbourhood', 'area / neighborhood / not'),
    ('location', 8): ('location not ideal (general)', 'location / location not / not'),
    ('location', 9): ('long walk', 'walk / walking / not'),
    # --- staff ---
    ('staff', 0): ('unfriendly, unhelpful staff', 'staff / not / service'),
    ('staff', 1): ('restaurant & bar service', 'breakfast / staff / restaurant'),
    ('staff', 2): ('housekeeping', 'cleaning / housekeeping / room'),
    ('staff', 3): ('check-in & room access', 'room / check / staff'),
    ('staff', 4): ('unhelpful reception / front desk', 'reception / reception staff / no'),
    ('staff', 5): ('atmosphere & other guests', 'guests / not / guest'),
    ('staff', 6): ('booking & payment disputes', 'booking / com / booking com'),
    ('staff', 7): ('no response to requests', 'no / asked / answer'),
    ('staff', 8): ('unhelpful reception / front desk', 'desk / desk staff / staff'),
    ('staff', 9): ('language barrier', 'english / speak / speak english'),
    # --- cleanliness ---
    ('cleanliness', 0): ('room not cleaned', 'room / not / cleaning'),
    ('cleanliness', 1): ('dirty bathroom & mould', 'bathroom / shower / toilet'),
    ('cleanliness', 2): ('cleaning could be better', 'cleaning / clean / cleanliness'),
    ('cleanliness', 3): ('dust & dirty floors', 'floor / dust / dusty'),
    ('cleanliness', 4): ('bad smell', 'smell / room / smelled'),
    ('cleanliness', 5): ('dirty sheets & stains', 'bed / sheets / stains'),
    ('cleanliness', 6): ('dirty, worn carpets', 'carpet / carpets / old'),
    ('cleanliness', 7): ('insects & pests', 'ants / room / cockroaches'),
    ('cleanliness', 8): ('towels not changed', 'towels / towel / clean towels'),
    ('cleanliness', 9): ('dirty pool / beach', 'pool / pool not / beach'),
    # --- facilities ---
    ('facilities', 0): ('no fridge / kitchen / microwave', 'fridge / kitchen / room'),
    ('facilities', 1): ('air conditioning', 'air / conditioning / air conditioning'),
    ('facilities', 2): ('no towels / soap', 'towels / soap / no'),
    ('facilities', 3): ('no kettle / coffee', 'coffee / tea / kettle'),
    ('facilities', 4): ('TV not working', 'tv / channels / work'),
    ('facilities', 5): ('little clothes storage', 'clothes / space / luggage'),
    ('facilities', 6): ('spa, gym & pool', 'spa / gym / pool'),
    ('facilities', 7): ('few power sockets', 'power / sockets / plug'),
    ('facilities', 8): ('drinking & hot water', 'water / hot water / bottles'),
    ('facilities', 9): ('lamps & lighting', 'light / lamp / lights'),
    # --- comfort ---
    ('comfort', 0): ('uncomfortable bed', 'bed / comfortable / uncomfortable'),
    ('comfort', 1): ('small, cramped room', 'small / rooms / room'),
    ('comfort', 2): ('small, cramped room', 'room / room small / small'),
    ('comfort', 3): ('bed too small for two', 'bed / beds / double'),
    ('comfort', 4): ('pillows', 'pillows / pillow / hard'),
    ('comfort', 5): ('small bathroom', 'bathroom / shower / bathroom small'),
    ('comfort', 6): ('uncomfortable (general)', 'uncomfortable / comfortable / not'),
    ('comfort', 7): ('sofa bed', 'sofa / sofa bed / couch'),
    ('comfort', 8): ('very small (unspecified)', 'small / thing / small thing'),
    ('comfort', 9): ('no space for luggage', 'luggage / suitcases / space'),
    # --- breakfast ---
    ('breakfast', 0): ('limited / poor breakfast', 'breakfast / not / options'),
    ('breakfast', 1): ('dinner & menu', 'food / dinner / menu'),
    ('breakfast', 2): ('breakfast room & area', 'room / breakfast / not'),
    ('breakfast', 3): ('expensive / not included', 'breakfast / expensive / included'),
    ('breakfast', 4): ('breakfast times / unavailable', 'breakfast / no breakfast / no'),
    ('breakfast', 5): ('coffee & tea', 'coffee / tea / milk'),
    ('breakfast', 6): ('eggs & bacon', 'eggs / bacon / scrambled'),
    ('breakfast', 7): ('little fresh fruit', 'fruit / fresh / fresh fruit'),
    ('breakfast', 8): ('buffet', 'buffet / breakfast buffet / buffet breakfast'),
    ('breakfast', 9): ('cold food', 'hot / cold / food'),
    # --- bathroom ---
    ('bathroom', 0): ('shower problems', 'shower / bath / not'),
    ('bathroom', 1): ('small / dirty bathroom', 'bathroom / toilet / small'),
    ('bathroom', 2): ('no hot water', 'hot / water / hot water'),
    ('bathroom', 3): ('towels & hooks', 'towels / towel / hang'),
    ('bathroom', 4): ('smell & extractor fan', 'smell / bathroom / fan'),
    ('bathroom', 5): ('no soap / shampoo', 'soap / shampoo / gel'),
    ('bathroom', 6): ('low water pressure', 'pressure / water pressure / water'),
    ('bathroom', 7): ('bathroom door & privacy', 'door / bathroom door / bathroom'),
    ('bathroom', 8): ('mirror & lighting', 'mirror / light / lighting'),
    ('bathroom', 9): ('sink & drain', 'sink / bathroom sink / wash'),
    # --- noise ---
    ('noise', 0): ('hearing the neighbours', 'room / hear / noisy'),
    ('noise', 1): ('night noise & music', 'night / loud / music'),
    ('noise', 2): ('traffic & trains', 'road / noise / traffic'),
    ('noise', 3): ('noise via windows & doors', 'windows / window / open'),
    ('noise', 4): ('noisy (unspecified)', 'noisy / noise / loud'),
    ('noise', 5): ('thin walls / no soundproofing', 'sound / not / soundproofing'),
    ('noise', 6): ('air-conditioning noise', 'air / conditioning / air conditioning'),
    ('noise', 7): ('creaky beds & floors', 'bed / beds / creaky'),
    ('noise', 8): ('construction work', 'construction / building / going'),
    ('noise', 9): ('thin walls / no soundproofing', 'walls / paper / walls paper'),
    # --- bed ---
    ('bed', 0): ('uncomfortable mattress / bed', 'mattress / bed / beds'),
    ('bed', 1): ('bed too small', 'bed / small / double'),
    ('bed', 2): ('pillows', 'pillows / pillow / soft'),
    ('bed', 3): ('dirty sheets', 'sheets / bed / shower'),
    ('bed', 4): ('sofa bed', 'sofa / sofa bed / bed'),
    ('bed', 5): ('uncomfortable mattress / bed', 'not / beds / comfortable'),
    ('bed', 6): ('hard to sleep (light, noise)', 'sleep / light / sleeping'),
    ('bed', 7): ('squeaky, noisy bed', 'noisy / noise / squeaky'),
    ('bed', 8): ('uncomfortable mattress / bed', 'bed hard / hard / bed'),
    # --- reception ---
    ('reception', 0): ('booking & room mix-ups', 'booking / room / not'),
    ('reception', 1): ('unhelpful reception / front desk', 'reception / staff / reception staff'),
    ('reception', 2): ('check-in time & delays', 'check / time / check time'),
    ('reception', 3): ('key, card & code problems', 'key / keys / code'),
    ('reception', 4): ('unhelpful reception / front desk', 'desk / not / staff'),
    ('reception', 5): ('requests not handled', 'breakfast / room / reception'),
    ('reception', 6): ('luggage storage', 'luggage / bags / airport'),
    ('reception', 7): ('pre-arrival communication', 'email / emails / arrival'),
    ('reception', 8): ('unwelcoming check-in staff', 'staff / check / not'),
    ('reception', 9): ('rude receptionist', 'receptionist / english / not'),
    # --- value_for_money ---
    ('value_for_money', 0): ('room overpriced', 'room / price / expensive'),
    ('value_for_money', 1): ('food & breakfast overpriced', 'breakfast / food / expensive'),
    ('value_for_money', 2): ('high price (general)', 'price / expensive / high'),
    ('value_for_money', 3): ('extra charges & deposit', 'booking / pay / charged'),
    ('value_for_money', 4): ('parking fee', 'parking / car / park'),
    ('value_for_money', 5): ('high price (general)', 'night / expensive night / 300'),
    ('value_for_money', 6): ('taxi / shuttle cost', 'taxi / airport / shuttle'),
    ('value_for_money', 7): ('drink prices', 'drinks / bar / wine'),
    ('value_for_money', 8): ('not worth its stars', 'star / not star / not'),
    ('value_for_money', 9): ('poor value for money', 'value / money / value money'),
    # --- view ---
    ('view', 0): ('no / poor view', 'view / room / window'),
    ('view', 1): ('no balcony', 'balcony / no balcony / chairs'),
    ('view', 2): ('no / poor view', 'view / no view / not'),
    ('view', 3): ('balcony door', 'door / balcony door / balcony'),
    ('view', 4): ('poor view from balcony', 'balcony / view balcony / balcony view'),
    ('view', 5): ('balcony privacy', 'privacy / balcony / curtain'),
    ('view', 6): ('terrace', 'terrace / rooftop terrace / terrace not'),
    ('view', 7): ('view among other complaints', 'breakfast / bathroom / no'),
    ('view', 8): ('noisy street-facing room', 'noisy / noise / loud'),
    # --- parking ---
    ('parking', 0): ('limited parking', 'parking / lot / tight'),
    ('parking', 1): ('limited parking', 'parking / lot / parking lot'),
    ('parking', 2): ('no parking', 'no / parking / no parking'),
    ('parking', 3): ('parking away / not included', 'parking / park / car'),
    ('parking', 4): ('expensive parking', 'expensive / parking / parking expensive'),
    ('parking', 5): ('paid parking', 'pay / paying / pay parking'),
    ('parking', 6): ('small or hard-to-find car park', 'car park / park / car'),
    ('parking', 7): ('carrying luggage from car park', 'luggage / stairs / elevator'),
    # --- pool ---
    ('pool', 0): ('small / closed pool', 'pool / small / swimming'),
    ('pool', 1): ('disappointing pool', 'pool / bar / room'),
    ('pool', 2): ('dirty pool', 'pool / clean / dirty'),
    ('pool', 3): ('cold pool', 'cold / heated / pool cold'),
    ('pool', 4): ('sun loungers', 'sun / chairs / loungers'),
    ('pool', 5): ('pool opening hours', 'open / early / closes'),
    ('pool', 6): ('pool towels', 'towels / pool towels / towels pool'),
    ('pool', 7): ('children & adults', 'kids / children / pool'),
    ('pool', 8): ('beach sunbeds', 'beach / beds / not'),
    ('pool', 9): ('loud music', 'music / loud / noisy'),
    # --- bar ---
    ('bar', 0): ('no bar / no atmosphere', 'bar / area / bar area'),
    ('bar', 1): ('bar closes early', 'closed / open / bar closed'),
    ('bar', 2): ('no bar or evening food', 'no / bar / food'),
    ('bar', 3): ('slow / poor bar service', 'staff / service / bar staff'),
    ('bar', 4): ('drink prices', 'drinks / drinks bar / expensive'),
    ('bar', 5): ('drink prices', 'prices / expensive / bar prices'),
    ('bar', 6): ('pool bar', 'pool / pool bar / beach'),
    ('bar', 7): ('no bar / no atmosphere', 'no bar / no / bar'),
    ('bar', 8): ('music & atmosphere', 'music / loud / noisy'),
    ('bar', 9): ('rooftop bar closed', 'rooftop / rooftop bar / roof'),
    # --- restaurant ---
    ('restaurant', 0): ('food not good', 'food / not / good'),
    ('restaurant', 1): ('restaurant closed', 'closed / restaurant closed / open'),
    ('restaurant', 2): ('limited dinner menu', 'menu / dinner / limited'),
    ('restaurant', 3): ('no proper restaurant', 'restaurant / not / no'),
    ('restaurant', 4): ('expensive food', 'expensive / food / restaurant'),
    ('restaurant', 5): ('slow / poor service', 'service / staff / restaurant'),
    ('restaurant', 6): ('no proper restaurant', 'no / no restaurant / restaurant'),
    ('restaurant', 7): ('few (vegetarian) options', 'food / options / food options'),
    ('restaurant', 8): ('cold food', 'cold / hot / food cold'),
    ('restaurant', 9): ('drinks', 'drinks / wine / cocktails'),
    # --- lift ---
    ('lift', 0): ('slow / broken lift', 'elevator / elevators / slow'),
    ('lift', 1): ('slow / broken lift', 'lift / lifts / slow'),
    ('lift', 2): ('steep stairs, no lift', 'stairs / steep / floor'),
    ('lift', 3): ('carrying luggage upstairs', 'luggage / carry / stairs'),
    ('lift', 4): ('no lift', 'elevator / floor / no elevator'),
    ('lift', 5): ('no lift', 'no elevator / no / elevator'),
    ('lift', 6): ('no lift', 'floor / lift / no lift'),
    ('lift', 7): ('no lift', 'no lift / lift / no'),
    ('lift', 8): ('lift noise', 'noisy / noise / room'),
    ('lift', 9): ('no lift', 'lift / lifts / car'),
    # --- wifi ---
    ('wifi', 0): ('slow wifi', 'wifi / slow / wi'),
    ('wifi', 1): ('weak wifi in the room', 'room / wifi / no'),
    ('wifi', 2): ('wifi not working', 'password / wifi / code'),
    ('wifi', 3): ('slow wifi', 'internet / internet connection / slow'),
    ('wifi', 4): ('wifi not working', 'work / wifi / working'),
    ('wifi', 5): ('wifi & TV / streaming', 'tv / netflix / watch'),
    ('wifi', 6): ('wifi only in lobby / paid', 'wifi / not / booking'),
    ('wifi', 7): ('weak wifi in the room', 'internet / room / internet room'),
    ('wifi', 8): ('wifi not working', 'internet / no internet / internet not'),
    ('wifi', 9): ('wifi among other complaints', 'water / shower / hot'),
    # --- beach ---
    ('beach', 0): ('beach far / poor', 'beach / not / no'),
    ('beach', 1): ('beach noise at night', 'music / loud / beach'),
}


# HU/EN: dicséretek / praise
POSITIVE_TOPIC_NAMES: dict[tuple[str, int], tuple[str, str]] = {
    # --- room ---
    ('room', 0): ('spacious room', 'room / spacious / rooms'),
    ('room', 1): ('clean room', 'clean / room clean / room'),
    ('room', 2): ('spacious room', 'room / spacious / room spacious'),
    ('room', 3): ('clean & comfortable', 'comfortable / clean comfortable / clean'),
    ('room', 4): ('room + breakfast + staff (list)', 'breakfast / staff / friendly'),
    ('room', 5): ('cosy room, pleasant temperature', 'cozy / air / warm'),
    ('room', 6): ('spacious apartment', 'apartment / spacious apartment / apartment clean'),
    ('room', 7): ('comfortable bed', 'bed / bed comfortable / beds'),
    ('room', 8): ('kitchenette', 'kitchen / kitchenette / fridge'),
    ('room', 9): ('balcony / terrace', 'balcony / terrace / room balcony'),
    # --- location ---
    ('location', 0): ('great location', 'location / great / great location'),
    ('location', 1): ('great location', 'clean / location / room'),
    ('location', 2): ('near station / walkable', 'station / walk / train'),
    ('location', 3): ('restaurants & shops nearby', 'restaurants / shops / food'),
    ('location', 4): ('near the beach', 'beach / pool / close beach'),
    ('location', 5): ('central location', 'center / town / central'),
    ('location', 6): ('near the airport', 'airport / shuttle / close airport'),
    ('location', 7): ('quiet location', 'quiet / quiet location / area'),
    ('location', 8): ('location + staff (list)', 'staff / friendly / friendly staff'),
    ('location', 9): ('location + breakfast (list)', 'breakfast / breakfast good / good'),
    # --- staff ---
    ('staff', 0): ('friendly, helpful staff', 'staff / friendly staff / friendly'),
    ('staff', 1): ('friendly, helpful staff', 'staff / staff friendly / friendly'),
    ('staff', 2): ('staff + cleanliness (list)', 'clean / location / staff'),
    ('staff', 3): ('welcoming staff', 'friendly / welcome / welcoming'),
    ('staff', 4): ('breakfast & restaurant staff', 'breakfast / food / good'),
    ('staff', 5): ('helpful host', 'host / hosts / communication'),
    ('staff', 6): ('extra help & luggage', 'helped / help / luggage'),
    ('staff', 7): ('reception staff', 'reception / reception staff / receptionist'),
    ('staff', 8): ('owner', 'owner / owners / owner friendly'),
    ('staff', 9): ('service', 'service / great service / customer'),
    # --- cleanliness ---
    ('cleanliness', 0): ('very clean (general)', 'clean / location / cleanliness'),
    ('cleanliness', 1): ('clean room', 'room / rooms / room clean'),
    ('cleanliness', 2): ('clean + friendly staff (list)', 'staff / friendly / friendly staff'),
    ('cleanliness', 3): ('clean & comfortable', 'clean comfortable / comfortable / clean'),
    ('cleanliness', 4): ('clean hotel', 'clean / staff / good'),
    ('cleanliness', 5): ('clean + comfortable bed', 'bed / beds / comfortable'),
    ('cleanliness', 6): ('daily housekeeping & fresh towels', 'towels / day / cleaned'),
    ('cleanliness', 7): ('clean + breakfast (list)', 'breakfast / clean breakfast / good'),
    ('cleanliness', 8): ('clean bathroom', 'bathroom / shower / bathroom clean'),
    ('cleanliness', 9): ('clean apartment', 'apartment / apartment clean / clean apartment'),
    # --- facilities ---
    ('facilities', 0): ('good facilities (general)', 'facilities / amenities / clean'),
    ('facilities', 1): ('coffee machine & fridge', 'coffee / fridge / tea'),
    ('facilities', 2): ('well-equipped kitchen', 'kitchen / equipped / equipped kitchen'),
    ('facilities', 3): ('towels, laundry & toiletries', 'towels / laundry / washing'),
    ('facilities', 4): ('spa, pool & hot tub', 'spa / pool / tub'),
    ('facilities', 5): ('air conditioning', 'air / air conditioning / conditioning'),
    ('facilities', 6): ('garden, lounge & terrace', 'garden / lounge / terrace'),
    ('facilities', 7): ('everything needed', 'needed / need / clean needed'),
    ('facilities', 8): ('TV & streaming', 'tv / netflix / smart'),
    ('facilities', 9): ('bikes & scooters', 'bikes / bike / rent'),
    # --- comfort ---
    ('comfort', 0): ('comfortable rooms', 'comfortable / rooms / room'),
    ('comfort', 1): ('spacious room', 'room / room spacious / spacious'),
    ('comfort', 2): ('comfortable bed', 'bed / comfortable bed / beds'),
    ('comfort', 3): ('comfortable bed', 'bed / bed comfortable / beds'),
    ('comfort', 4): ('clean & comfortable', 'clean comfortable / clean / comfortable'),
    ('comfort', 5): ('bed + shower (list)', 'bathroom / shower / bed'),
    ('comfort', 6): ('clean, spacious room', 'clean / room / spacious room'),
    ('comfort', 7): ('spacious apartment', 'apartment / spacious apartment / apartment spacious'),
    ('comfort', 8): ('clean, spacious room', 'spacious / spacious clean / clean'),
    ('comfort', 9): ('comfort + breakfast (list)', 'breakfast / staff / good'),
    # --- breakfast ---
    ('breakfast', 0): ('great breakfast', 'breakfast / breakfast good / good'),
    ('breakfast', 1): ('great breakfast', 'breakfast / good breakfast / good'),
    ('breakfast', 2): ('location + room + breakfast (list)', 'room / location / breakfast'),
    ('breakfast', 3): ('food & dinner', 'food / dinner / food good'),
    ('breakfast', 4): ('staff + breakfast (list)', 'staff / friendly / helpful'),
    ('breakfast', 5): ('coffee & pastries', 'coffee / tea / fresh'),
    ('breakfast', 6): ('food & dinner', 'food / dinner / delicious'),
    ('breakfast', 7): ('buffet variety', 'buffet / buffet breakfast / breakfast buffet'),
    ('breakfast', 8): ('homemade bread & croissants', 'bread / croissants / homemade'),
    ('breakfast', 9): ('host + breakfast (list)', 'host / hosts / lovely'),
    # --- bathroom ---
    ('bathroom', 0): ('spacious bathroom', 'bathroom / room / spacious'),
    ('bathroom', 1): ('great shower', 'shower / room / great'),
    ('bathroom', 2): ('towels & toiletries', 'towels / toiletries / washing'),
    ('bathroom', 3): ('bed + shower (list)', 'bed / comfortable / beds'),
    ('bathroom', 4): ('great shower', 'shower / great shower / good shower'),
    ('bathroom', 5): ('clean bathroom', 'clean / bathroom clean / bathrooms'),
    ('bathroom', 6): ('bathtub', 'bath / tub / hot tub'),
    ('bathroom', 7): ('good water pressure', 'pressure / water pressure / water'),
    ('bathroom', 8): ('spacious bathroom', 'bathroom / bathroom nice / nice'),
    ('bathroom', 9): ('hot water', 'water / hot water / hot'),
    # --- noise ---
    ('noise', 0): ('quiet location', 'location / quiet / quiet location'),
    ('noise', 1): ('quiet room', 'room / quiet room / quiet'),
    ('noise', 2): ('quiet room', 'room / quiet / room quiet'),
    ('noise', 3): ('clean & quiet', 'clean / quiet / clean quiet'),
    ('noise', 4): ('no street noise', 'noise / no / no noise'),
    ('noise', 5): ('quiet at night', 'night / quiet night / sleep'),
    ('noise', 6): ('staff + quiet (list)', 'staff / breakfast / friendly'),
    ('noise', 7): ('quiet & comfortable', 'comfortable / quiet comfortable / comfortable quiet'),
    ('noise', 8): ('good soundproofing', 'sound / proofing / sound proofing'),
    ('noise', 9): ('pleasant sounds (river, waves)', 'music / river / sound'),
    # --- bed ---
    ('bed', 0): ('comfortable bed', 'bed / comfortable bed / comfortable'),
    ('bed', 1): ('comfortable bed', 'bed comfortable / comfortable / sleep'),
    ('bed', 2): ('room + bed (list)', 'room / bed / room clean'),
    ('bed', 3): ('clean, comfy bed linen', 'clean / clean comfortable / sheets'),
    ('bed', 4): ('room + bed (list)', 'room / rooms / spacious'),
    ('bed', 5): ('location + bed (list)', 'location / view / great'),
    ('bed', 6): ('shower + bed (list)', 'shower / bathroom / bed'),
    ('bed', 7): ('pillows', 'pillows / bed pillows / pillows comfortable'),
    ('bed', 8): ('breakfast + bed (list)', 'breakfast / bed breakfast / bed'),
    ('bed', 9): ('staff + bed (list)', 'staff / friendly / helpful'),
    # --- reception ---
    ('reception', 0): ('friendly reception staff', 'reception / staff / reception staff'),
    ('reception', 1): ('easy check-in', 'check / easy / easy check'),
    ('reception', 2): ('early check-in', 'early / check / late'),
    ('reception', 3): ('luggage storage', 'luggage / airport / bags'),
    ('reception', 4): ('friendly check-in staff', 'staff / check / friendly'),
    ('reception', 5): ('friendly reception staff', 'desk / desk staff / helpful'),
    ('reception', 6): ('warm welcome', 'welcome / host / arrival'),
    ('reception', 7): ('friendly reception staff', 'receptionist / receptionists / helpful'),
    ('reception', 8): ('self check-in & key instructions', 'key / whatsapp / keys'),
    # --- value_for_money ---
    ('value_for_money', 0): ('value for money', 'money / value money / value'),
    ('value_for_money', 1): ('good room price', 'room / price / good'),
    ('value_for_money', 2): ('location + value (list)', 'location / location price / price'),
    ('value_for_money', 3): ('breakfast included', 'breakfast / free breakfast / food'),
    ('value_for_money', 4): ('free parking', 'parking / free / free parking'),
    ('value_for_money', 5): ('clean + value (list)', 'clean / comfortable / clean good'),
    ('value_for_money', 6): ('value for money', 'value / good / good value'),
    ('value_for_money', 7): ('airport shuttle', 'airport / shuttle / free shuttle'),
    ('value_for_money', 8): ('staff + value (list)', 'staff / friendly / friendly staff'),
    ('value_for_money', 9): ('free coffee', 'free / coffee / free coffee'),
    # --- view ---
    ('view', 0): ('view from the room', 'room / view / views'),
    ('view', 1): ('balcony', 'balcony / room / view balcony'),
    ('view', 2): ('sea view', 'sea / sea view / ocean'),
    ('view', 3): ('city / mountain view', 'location / view / views'),
    ('view', 4): ('amazing view', 'view / views / amazing'),
    ('view', 5): ('terrace / rooftop', 'terrace / rooftop / roof'),
    ('view', 6): ('amazing view', 'views / view / beautiful'),
    ('view', 7): ('breakfast with a view', 'breakfast / view / delicious'),
    ('view', 8): ('lake & sunset', 'lake / sunset / view lake'),
    ('view', 9): ('pool view', 'pool / pool view / view pool'),
    # --- parking ---
    ('parking', 0): ('parking available', 'parking / location / car'),
    ('parking', 1): ('convenient parking', 'parking / car / park'),
    ('parking', 2): ('free parking', 'free / free parking / parking'),
    ('parking', 3): ('on-site parking', 'parking / free / car'),
    ('parking', 4): ('room + parking (list)', 'room / clean / apartment'),
    ('parking', 5): ('secure parking', 'secure / secure parking / safe'),
    ('parking', 6): ('breakfast + parking (list)', 'breakfast / good / parking'),
    ('parking', 7): ('easy parking', 'easy / easy parking / easy park'),
    ('parking', 8): ('staff + parking (list)', 'staff / friendly / friendly staff'),
    ('parking', 9): ('on-site parking', 'site / parking site / site parking'),
    # --- pool ---
    ('pool', 0): ('nice pool', 'pool / swimming / nice'),
    ('pool', 1): ('great pool for children', 'pool / kids / loved'),
    ('pool', 2): ('room + pool (list)', 'room / pool / comfortable'),
    ('pool', 3): ('rooftop pool with a view', 'view / rooftop / rooftop pool'),
    ('pool', 4): ('pool bar & restaurant', 'bar / restaurant / pool bar'),
    ('pool', 5): ('breakfast + pool (list)', 'breakfast / pool breakfast / breakfast pool'),
    ('pool', 6): ('clean pool', 'clean / pool clean / pool'),
    ('pool', 7): ('beach + pool (list)', 'beach / towels / pool beach'),
    ('pool', 8): ('heated pool', 'heated / heated pool / warm'),
    ('pool', 9): ('staff + pool (list)', 'staff / friendly / helpful'),
    # --- bar ---
    ('bar', 0): ('restaurant & bar', 'restaurant / bar / bar restaurant'),
    ('bar', 1): ('bar food & live music', 'bar / restaurant / food'),
    ('bar', 2): ('nice bar area', 'bar / area / bar area'),
    ('bar', 3): ('drinks & cocktails', 'drinks / cocktails / good'),
    ('bar', 4): ('bars nearby', 'bars / restaurants / bars restaurants'),
    ('bar', 5): ('pool bar', 'pool / pool bar / bar'),
    ('bar', 6): ('bar staff', 'staff / bar staff / friendly'),
    ('bar', 7): ('rooftop bar', 'rooftop / rooftop bar / roof'),
    ('bar', 8): ('breakfast + bar (list)', 'breakfast / bar / breakfast bar'),
    ('bar', 9): ('pub', 'pub / food / food pub'),
    # --- restaurant ---
    ('restaurant', 0): ('restaurants & dining', 'restaurant / restaurants / dining'),
    ('restaurant', 1): ('great food', 'food / food good / good'),
    ('restaurant', 2): ('restaurant food', 'restaurant / bar / good'),
    ('restaurant', 3): ('great food', 'food / good / good food'),
    ('restaurant', 4): ('dinner', 'dinner / ate / evening'),
    ('restaurant', 5): ('on-site restaurant', 'restaurant / open / food'),
    ('restaurant', 6): ('restaurant staff', 'staff / friendly / restaurant staff'),
    ('restaurant', 7): ('beach + restaurants (list)', 'beach / restaurants / pool'),
    ('restaurant', 8): ('breakfast restaurant', 'breakfast / restaurant / restaurant breakfast'),
    ('restaurant', 9): ('rooftop restaurant', 'rooftop / terrace / rooftop restaurant'),
    # --- lift ---
    ('lift', 0): ('has a lift', 'elevator / elevators / room'),
    ('lift', 1): ('has a lift', 'lift / lifts / stairs'),
    # --- wifi ---
    ('wifi', 0): ('room + wifi (list)', 'room / good / wifi'),
    ('wifi', 1): ('good wifi', 'good wifi / good / wifi'),
    ('wifi', 2): ('good wifi', 'wifi / worked / wifi good'),
    ('wifi', 3): ('fast internet', 'internet / good internet / fast internet'),
    ('wifi', 4): ('free wifi', 'free / free wifi / free wi'),
    ('wifi', 5): ('wifi + TV (list)', 'tv / netflix / channels'),
    ('wifi', 6): ('fast internet', 'fast / fast wifi / wifi'),
    ('wifi', 7): ('fast internet', 'strong / strong wifi / wifi strong'),
    ('wifi', 8): ('wifi + air conditioning (list)', 'ac / air / worked'),
    ('wifi', 9): ('breakfast + wifi (list)', 'breakfast / good breakfast / breakfast good'),
    # --- beach ---
    ('beach', 0): ('beach location', 'beach / location / great'),
    ('beach', 1): ('close to the beach', 'beach / close / close beach'),
    ('beach', 2): ('pool + staff + beach (list)', 'pool / staff / nice'),
    ('beach', 3): ('sea view', 'sea / view / sea view'),
    ('beach', 4): ('short walk to the beach', 'walk / walk beach / beach'),
    ('beach', 5): ('beach & restaurants nearby', 'restaurants / close / beach'),
    ('beach', 6): ('beach & restaurants nearby', 'walk / restaurants / walking'),
    ('beach', 7): ('close to the beach', 'beach / located / beautiful'),
    ('beach', 8): ('breakfast with a sea view', 'breakfast / restaurant / view'),
    ('beach', 9): ('clean beach', 'clean / towels / beach'),
}


TOPIC_NAMES = {"negative": NEGATIVE_TOPIC_NAMES, "positive": POSITIVE_TOPIC_NAMES}


def get_topic_names(sentiment: str, topic_table: pd.DataFrame | None = None) -> dict[tuple[str, int], str]:
    """
    HU: {(aspektus, téma): név} a rename_topics() számára. Ha topic_table is adott, ellenőrzi az
        ujjlenyomatot (automatikus címke); eltérés esetén hibát dob, mert akkor a fürtök megváltoztak.
    EN: {(aspect, topic): name} for rename_topics(). If topic_table is given, checks the fingerprint
        (automatic label); raises on mismatch, because then the clusters have changed.
    """
    entries = TOPIC_NAMES[sentiment]
    if topic_table is not None:
        auto = {(a, int(t)): lab for a, t, lab in
                topic_table[["aspect", "topic", "label"]].itertuples(index=False)}
        mismatched = [(key, expected, auto.get(key)) for key, (_, expected) in entries.items()
                      if auto.get(key) != expected]
        unnamed = sorted(set(auto) - set(entries))
        if mismatched or unnamed:
            lines = [f"  {k}: expected {e!r}, got {g!r}" for k, e, g in mismatched[:10]]
            raise ValueError(
                f"Topic names ({TOPIC_VERSION}) do not match this topic table: {len(mismatched)} label "
                f"mismatches, {len(unnamed)} unnamed topics. The clusters changed — re-do the names.\n"
                + "\n".join(lines))
    return {key: name for key, (name, _) in entries.items()}
