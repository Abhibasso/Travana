import pickle

import nltk
from nltk.corpus import stopwords

tokenizer = nltk.RegexpTokenizer(r'\w+')
lemma = nltk.wordnet.WordNetLemmatizer()
stopw = stopwords.words("english")

pred_post = 'that great can you do something about how you aggregate values opposed to what you are built on london traffic yes how are you guys would be great to see you an alternative to alternative facts belfast this afternoon and one of its greatest exports congratulationson your new grant from brilliant work and well deserved very worrying to see congressional republications trying to kill anti corruption rule pwyp uk members urge pm theresa may to defend oil transparency anti corruption law in the u this week prime minister may please discuss international development with president trump gates foundation research can t be published in top journals the women leading africa open data drive trump team prepares for cuts nice to meet you today in the shuttle van would be great to talk about g growth and the demographic dividend in ssa d in in sub saharan african children will live in poverty of global poverty off to davos for a strategy session on how to expand internet access to people living in poverty more here i had just sent you an email which will land in your inbox as soon as i get wifi access would be great to catch up echoes of amazon echo microphone always on off to dc to spend some time planning for with my brilliant team members there the perfect sandwich h t really excellent podcast from jim o neill on globalisation great advice for junior colleagues nervous about contributing in meetings if you have a good point make it i can identify with this work travel not nearly as exciting or enjoyable as it sounds good thoughts from massive development conferences worthwhile ora massive drain of cash demographic detriment looking to the uk in very interesting piece on j bach obsession with god and how it comes out in his music check this out js bach on analog synthesiser inspired byand in an effort to avoid my own echo chamber i have unfollowed everyone and am starting again proudly reading about my grandfather role in the arctic convoys amazing bravery at just years old years ago today jimi hendrix wrote purple haze waiting to go on stage for a boxing day gig in declines in poverty amp mortality expansion of renewable energy amp rights reasons to be thankful in angus deaton link between us white male mortality experience of physical pain and trump voting is striking'

def tokenize(post):
    tokenized_df = tokenizer.tokenize(post)
    word_bag = [lemma.lemmatize(token.lower()) for token in tokenized_df if token not in stopw and not token.isdigit()]
    return word_bag


def find_features(document):
    words = set(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)
    return features


with open('classifier.pickle', 'rb') as f:
    clf = pickle.load(f)

with open('word_features.pickle', 'rb') as f:
    word_features = pickle.load(f)

tokenize_words = tokenize(pred_post)
features = find_features(tokenize_words)
mbti_type = clf.classify(features)
if mbti_type.startswith('e'):
    profile_type = 'Adventurous'
else:
    profile_type = 'Artistic'

print("Profile Type : ", profile_type)
print("MBTI type : ", mbti_type)
