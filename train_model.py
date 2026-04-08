import os
import re
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, classification_report

# --- NLTK Setup (graceful fallback) ---
STOP_WORDS = {"a", "an", "the", "and", "is", "in", "it", "to", "of", "for",
              "on", "that", "this", "with", "was", "are", "be", "has", "had",
              "not", "but", "or", "at", "by", "from", "as", "do", "if", "no",
              "he", "she", "they", "we", "you", "i", "me", "my", "your",
              "his", "her", "its", "our", "their", "what", "which", "who",
              "when", "where", "how", "all", "each", "every", "both", "few",
              "more", "most", "other", "some", "such", "than", "too", "very",
              "can", "will", "just", "should", "now", "also", "into", "only",
              "about", "up", "out", "so", "him", "them", "then", "these",
              "those", "been", "have", "would", "could", "did", "does",
              "said", "were", "over", "after", "before", "between", "own",
              "same", "because", "while", "during", "through"}
try:
    import nltk
    from nltk.corpus import stopwords
    nltk.download("stopwords", quiet=True)
    STOP_WORDS = set(stopwords.words("english"))
except Exception:
    pass

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def preprocess_text(text: str) -> str:
    """Clean and tokenize input text, removing stop words."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r"[^a-zA-Z\s]", "", text).lower()
    tokens = [w for w in text.split() if w not in STOP_WORDS]
    return " ".join(tokens)


def generate_training_data() -> str:
    """Generate a comprehensive, realistic training dataset for fake news detection."""
    real_news = [
        # --- Politics & Government (Real) ---
        "The Senate passed a bipartisan infrastructure bill with a vote of 69 to 30 after months of negotiations between Democrats and Republicans.",
        "The Federal Reserve announced it would raise interest rates by 0.25 percentage points, citing persistent inflation and a strong labor market.",
        "President signed an executive order aimed at reducing prescription drug prices for Medicare recipients by allowing negotiation with pharmaceutical companies.",
        "The Supreme Court ruled 6-3 that employers cannot discriminate based on sexual orientation or gender identity under Title VII of the Civil Rights Act.",
        "The Department of Labor reported that the economy added 263,000 jobs in November, exceeding economists' expectations of 200,000 new positions.",
        "Congress approved a $1.7 trillion spending bill to fund the government through September, averting a shutdown days before the deadline.",
        "The Environmental Protection Agency finalized new rules limiting methane emissions from oil and gas operations across the country.",
        "The House passed legislation to codify same-sex marriage protections with bipartisan support, sending the bill to the President's desk.",
        "NATO allies agreed to increase defense spending targets during an emergency summit in response to growing geopolitical tensions.",
        "The Federal Aviation Administration grounded all domestic flights temporarily after a system outage affected the Notice to Air Missions database.",
        "The Census Bureau reported that the U.S. population grew by 0.4 percent last year, the slowest rate of growth since the nation's founding.",
        "The Department of Justice appointed a special counsel to oversee investigations into the former president's handling of classified documents.",
        "The Treasury Department imposed new sanctions on Russian oligarchs and entities linked to the country's defense sector.",
        "The World Health Organization declared that COVID-19 no longer constitutes a public health emergency of international concern.",
        "The United Nations General Assembly voted overwhelmingly to condemn the invasion and demand the withdrawal of military forces.",
        
        # --- Science & Technology (Real) ---
        "NASA's James Webb Space Telescope captured its first full-color image of deep space, revealing thousands of galaxies in unprecedented detail.",
        "Researchers at MIT developed a new desalination system powered by solar energy that can produce drinking water at a fraction of current costs.",
        "The FDA approved a new Alzheimer's drug after clinical trials showed it slowed cognitive decline by 27 percent over 18 months.",
        "SpaceX successfully launched and landed a Starship prototype for the first time, marking a major milestone in reusable rocket technology.",
        "A study published in Nature found that global temperatures have risen 1.1 degrees Celsius above pre-industrial levels due to greenhouse gas emissions.",
        "Scientists at CERN announced the discovery of a new particle consistent with predictions from the Standard Model of physics.",
        "Google's DeepMind AI system AlphaFold predicted the 3D structure of nearly every known protein, revolutionizing biological research.",
        "Electric vehicle sales surpassed 10 million units globally for the first time, driven by falling battery costs and government incentives.",
        "Researchers successfully used CRISPR gene editing to treat sickle cell disease in clinical trials, with patients showing lasting improvements.",
        "The World Meteorological Organization confirmed that the past eight years were the warmest on record globally.",
        "Apple announced quarterly revenue of $117 billion, driven by strong iPhone sales and growth in its services division.",
        "Microsoft acquired the gaming company for $68.7 billion, making it the largest deal in gaming industry history.",
        "A new study in The Lancet found that regular exercise reduces the risk of developing depression by up to 30 percent.",
        "The European Space Agency launched a new satellite to monitor sea level changes and ice sheet dynamics in unprecedented detail.",
        "Pfizer reported that its updated COVID-19 booster shot was 73 percent effective at preventing hospitalization from the latest variant.",
        
        # --- Business & Economy (Real) ---
        "Amazon reported a 9 percent increase in quarterly revenue, driven by growth in cloud computing and advertising services.",
        "The unemployment rate fell to 3.4 percent in January, the lowest level in more than 50 years according to Bureau of Labor Statistics data.",
        "Inflation slowed to 4 percent in May, down from a peak of 9.1 percent last June, according to the Consumer Price Index report.",
        "The stock market ended the year up 24 percent, with technology stocks leading gains amid enthusiasm about artificial intelligence.",
        "Bank regulators seized a major regional bank after a run on deposits, marking the second-largest bank failure in U.S. history.",
        "Global oil prices dropped below $70 per barrel amid concerns about weakening demand from China and rising U.S. production.",
        "The housing market saw existing home sales fall to their lowest level in 13 years as mortgage rates remained above 7 percent.",
        "Tesla delivered 1.3 million vehicles in the quarter, a new record, though growth slowed compared to previous years.",
        "The International Monetary Fund projected global economic growth of 3.0 percent for the year, citing resilient consumer spending.",
        "Walmart announced it would raise its minimum wage to $14 per hour, affecting approximately 340,000 store workers.",
        
        # --- Health (Real) ---
        "The Centers for Disease Control reported that life expectancy in the United States increased by 1.1 years following pandemic-era declines.",
        "A large-scale clinical trial found that a Mediterranean diet supplemented with olive oil reduced the risk of cardiovascular events by 30 percent.",
        "Hospitals across the country reported a surge in respiratory illness cases driven by influenza, RSV, and COVID-19 occurring simultaneously.",
        "The FDA issued a warning about contaminated eye drop products after reports of serious infections and one death linked to the products.",
        "Researchers published evidence that ultra-processed foods are linked to increased risk of cancer, heart disease, and premature death.",
        
        # --- Sports (Real) ---
        "Argentina won the FIFA World Cup in a dramatic penalty shootout against France after the match ended 3-3 after extra time.",
        "The Kansas City Chiefs won the Super Bowl with a last-second field goal, defeating the Philadelphia Eagles 38 to 35.",
        "Lionel Messi signed with Inter Miami in Major League Soccer after leaving Paris Saint-Germain, sparking record ticket demand.",
        "The International Olympic Committee announced that the 2032 Summer Olympics would be held in Brisbane, Australia.",
        "A major league baseball player hit his 62nd home run of the season, breaking the American League single-season record.",
        
        # --- World News (Real) ---
        "A devastating earthquake struck Turkey and Syria, killing more than 50,000 people and displacing millions from their homes.",
        "Wildfires in Canada burned more than 45 million acres, producing smoke that affected air quality across the northeastern United States.",
        "The European Union agreed to a landmark artificial intelligence regulation framework, the first comprehensive AI law in the world.",
        "India's population surpassed China's to become the world's most populous country, according to United Nations estimates.",
        "A submersible vessel imploded during a dive to the Titanic wreck site, killing all five passengers on board.",
    ]

    fake_news = [
        # --- Health Misinformation ---
        "EXPOSED: Vaccines contain microchips designed by Bill Gates to track your every movement through 5G satellite networks.",
        "URGENT: Doctors confirm that drinking bleach mixed with lemon juice cures cancer within 48 hours, Big Pharma terrified.",
        "SECRET REVEALED: All COVID-19 vaccines are actually designed to alter human DNA and create a race of obedient super-soldiers.",
        "BREAKING: Scientists prove that 5G cell towers cause brain tumors and the government is covering it up to protect telecom profits.",
        "SHOCKING: Hospitals caught injecting patients with tracking nanobots disguised as flu shots to monitor their brain activity.",
        "BOMBSHELL: Eating only raw meat and drinking unpasteurized milk eliminates all diseases, medical establishment panics.",
        "SUPPRESSED STUDY: Water fluoridation proven to lower IQ by 40 points, government deliberately dumbing down population.",
        "CONFIRMED: The FDA is secretly adding mind-control chemicals to all prescription medications sold in the United States.",
        "WHISTLEBLOWER: All cancer treatments are deliberately designed to fail so hospitals can keep billing patients indefinitely.",
        "BREAKING: Essential oils proven more effective than chemotherapy by underground research team suppressed by Big Pharma.",
        
        # --- Political Conspiracy ---
        "EXCLUSIVE: Secret documents prove the moon landing was filmed in a Hollywood studio by Stanley Kubrick under CIA direction.",
        "REVEALED: The deep state is using weather manipulation technology to create hurricanes and target conservative states.",
        "BOMBSHELL: Leaked emails prove that all elections since 2000 have been rigged by a shadow government of international bankers.",
        "CONFIRMED: The government has been hiding proof of alien contact for 70 years and uses alien technology for military weapons.",
        "SHOCKING: World leaders caught on hidden camera planning to depopulate Earth by 80 percent using engineered food shortages.",
        "SECRET PLOT: The United Nations is planning to abolish all national borders and create a single world government by next year.",
        "EXPOSED: All major news networks are controlled by a single family that decides what stories to cover and which to suppress.",
        "BREAKING: Former CIA agent reveals that the government manufactured the pandemic to impose permanent martial law worldwide.",
        "LEAKED: Billionaires have already built underground cities to survive the apocalypse they are deliberately engineering.",
        "URGENT: Global elites planning to replace all currency with a single digital coin they control to enslave the population.",
        
        # --- Science Misinformation ---
        "PROVEN: The Earth is actually flat and NASA has been faking satellite images for decades to maintain the globe conspiracy.",
        "BREAKING: Scientists discover that the sun is actually cold and made of ice, entire field of astrophysics proven wrong.",
        "SHOCKING: Gravity is not real — objects fall because the flat Earth accelerates upward at 9.8 meters per second squared.",
        "CONFIRMED: Dinosaurs never existed. Fossils were planted by scientists to justify evolution and undermine religious teachings.",
        "EXPOSED: Climate change is a complete hoax invented by China to weaken American manufacturing and steal factory jobs.",
        "BOMBSHELL: Research proves that crystals can heal any disease by realigning the body's quantum energy frequencies.",
        "SUPPRESSED: Free energy devices have been invented dozens of times but oil companies murder the inventors every time.",
        "BREAKING: The Large Hadron Collider has opened a portal to another dimension and demons are now entering our world.",
        "REVEALED: All species were created simultaneously 6,000 years ago and evolution is a fraud created to deny divine creation.",
        "URGENT: WiFi radiation proven to cause permanent infertility in humans, tech companies suppressing all research.",
        
        # --- Celebrity & Entertainment Fake News ---
        "EXCLUSIVE: Famous pop star confirmed to be a cloned robot replacement after the real person died in a secret ritual sacrifice.",
        "SHOCKING: Hollywood actor reveals all celebrities are actually shape-shifting reptilian aliens from the constellation Draco.",
        "CONFIRMED: Major streaming platform caught inserting subliminal messages in shows to brainwash viewers into political obedience.",
        "BREAKING: Pop star's death was actually faked and she is living on a secret island with other celebrities who staged their deaths.",
        "EXPOSED: All Grammy and Oscar winners are pre-selected by the Illuminati five years in advance as part of their agenda.",
        
        # --- Economic Misinformation ---
        "BREAKING: The US dollar will be completely worthless by next Monday. Buy gold immediately before the total economic collapse.",
        "EXCLUSIVE: The Federal Reserve is secretly printing trillions of dollars and giving it directly to a small group of billionaires.",
        "CONFIRMED: All major banks will freeze customer accounts next week as part of a planned global financial reset operation.",
        "SHOCKING: Bitcoin was actually created by the NSA as a tool to track all financial transactions and monitor citizens.",
        "URGENT: The IRS has been dissolved and all Americans no longer need to pay taxes starting immediately, spread the word.",
        
        # --- Social Media Hoaxes ---
        "WARNING: If you don't share this post with 20 people your social media account will be deleted permanently tomorrow morning.",
        "ALERT: Hackers have released a virus that steals your banking information if you open any message from an unknown number.",
        "BREAKING: The government is monitoring all private messages on social media and will arrest anyone who criticizes the president.",
        "CONFIRMED: A new law makes it illegal to post memes online and violators will face up to 5 years in federal prison.",
        "URGENT: Your phone is listening to every conversation you have and selling transcripts to the government and advertisers.",
        "SHOCKING: Drinking colloidal silver turns your immune system invincible and pharmaceutical companies want to ban it.",
        "EXCLUSIVE: Interview with time traveler from 2089 confirms World War 3 starts next month over control of Antarctica resources.",
        "REVEALED: Chemtrails in the sky are actually chemicals designed to suppress free will and make citizens accept authoritarian rule.",
        "BREAKING: Major world leader secretly replaced by body double three years ago after assassination, close associates confirm.",
        "CONFIRMED: Underground tunnels connect all major cities and are used by elites for secret transportation and illegal activities.",
    ]

    data = []
    for text in real_news:
        data.append({"text": text, "label": 1})
    for text in fake_news:
        data.append({"text": text, "label": 0})

    # Augment data by creating slight variations
    augmented = []
    for item in data:
        augmented.append(item)
        # Add a version with minor prefix/suffix changes
        augmented.append({"text": "Report: " + item["text"], "label": item["label"]})
        augmented.append({"text": item["text"] + " Sources confirm this story.", "label": item["label"]})

    csv_path = os.path.join(BASE_DIR, "news_dataset.csv")
    pd.DataFrame(augmented).to_csv(csv_path, index=False)
    print(f"Generated {len(augmented)} training samples ({len(real_news)} real, {len(fake_news)} fake, 3x augmented)")
    return csv_path


def train():
    """Train the PassiveAggressive classifier and save to disk."""
    csv_file = generate_training_data()

    print("Loading dataset...")
    df = pd.read_csv(csv_file)
    df = df.dropna(subset=["text"])

    print("Preprocessing text...")
    df["clean_text"] = df["text"].apply(preprocess_text)

    print("Extracting features (TF-IDF)...")
    vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), max_df=0.9, min_df=2)
    X = vectorizer.fit_transform(df["clean_text"])
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print("Training PassiveAggressive classifier...")
    model = PassiveAggressiveClassifier(max_iter=100, C=0.5, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nModel Accuracy: {accuracy * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["FAKE", "REAL"]))

    joblib.dump(vectorizer, os.path.join(BASE_DIR, "vectorizer.pkl"))
    joblib.dump(model, os.path.join(BASE_DIR, "model.pkl"))
    print("Done! Saved model.pkl and vectorizer.pkl.")


if __name__ == "__main__":
    train()
