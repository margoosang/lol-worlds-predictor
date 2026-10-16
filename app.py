import base64
from pathlib import Path
import pandas as pd
import streamlit as st

st.set_page_config(page_title="LoL Worlds 2026 Predictor", page_icon="🏆", layout="wide", initial_sidebar_state="collapsed")

DATA_FOLDER = Path(__file__).resolve().parent
LOGO_FOLDER = DATA_FOLDER / "logos"

@st.cache_data
def load_ratings():
    return pd.read_csv(DATA_FOLDER / "team_ratings.csv").sort_values("elo", ascending=False).reset_index(drop=True)

ratings_df = load_ratings()
ratings = dict(zip(ratings_df["team"], ratings_df["elo"]))

# Keep these display names independent of the historical dataset's team identifiers.
worlds_teams = {
    "LCK": ["Gen.G", "Hanwha Life Esports", "T1", "Dplus Kia"],
    "LPL": ["Anyone's Legend", "Bilibili Gaming", "Top Esports", "Invictus Gaming"],
    "LEC": ["G2 Esports", "Movistar KOI", "Karmine Corp"],
    "LCS": ["Team Liquid", "LYON", "Cloud9"],
    "LCP": ["Team Secret Whales", "CTBC Flying Oyster", "MVK Esports"],
    "CBLOL": ["LOS", "FURIA"],
}

team_acronyms = {
    "Gen.G": "GEN", "Hanwha Life Esports": "HLE", "T1": "T1", "Dplus Kia": "DK",
    "Anyone's Legend": "AL", "Bilibili Gaming": "BLG", "Top Esports": "TES", "Invictus Gaming": "IG",
    "G2 Esports": "G2", "Movistar KOI": "MKOI", "Karmine Corp": "KC",
    "Team Liquid": "TLAW", "LYON": "LYON", "Cloud9": "C9",
    "Team Secret Whales": "TSW", "CTBC Flying Oyster": "CFO", "MVK Esports": "MVK",
    "LOS": "LOS", "FURIA": "FUR",
}

team_aliases = {
    "Gen.G": ["Gen.G", "Gen.G Esports"],
    "Dplus Kia": ["Dplus Kia", "Dplus KIA", "Dplus"],
    "Hanwha Life Esports": ["Hanwha Life Esports"],
    "Team Secret Whales": ["Team Secret Whales", "Secret Whales"],
    "MVK Esports": ["MVK Esports", "MGN Vikings Esports"],
    "LOS": ["LOS", "LØS", "LÃ˜S"],
}

worlds_ratings = {}
team_regions = {}
missing_teams = []
for region, region_teams in worlds_teams.items():
    for team in region_teams:
        matched = next((alias for alias in team_aliases.get(team, [team]) if alias in ratings), None)
        if matched is None:
            missing_teams.append(team)
        else:
            worlds_ratings[team] = ratings[matched]
            team_regions[team] = region

teams = sorted(worlds_ratings)

def acronym(team):
    return team_acronyms.get(team, team)

def team_label(team):
    return acronym(team)

def get_team_logo(team):
    path = LOGO_FOLDER / f"{acronym(team)}.png"
    return path if path.is_file() else None

@st.cache_data
def encode_logo(path):
    if path is None:
        return None
    encoded = base64.b64encode(Path(path).read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"

def get_logo_data_url(team):
    return encode_logo(get_team_logo(team))

def expected_score(rating_a, rating_b):
    return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))

def game_probability(team_a, team_b):
    return expected_score(worlds_ratings[team_a], worlds_ratings[team_b])

def score_probabilities(p, best_of):
    """Exact terminating score probabilities; first number is Team A's wins."""
    q = 1 - p
    if best_of == 1:
        return {(1, 0): p, (0, 1): q}
    if best_of == 3:
        return {(2, 0): p**2, (2, 1): 2*p**2*q,
                (0, 2): q**2, (1, 2): 2*q**2*p}
    if best_of == 5:
        return {(3, 0): p**3, (3, 1): 3*p**3*q, (3, 2): 6*p**3*q**2,
                (0, 3): q**3, (1, 3): 3*q**3*p, (2, 3): 6*q**3*p**2}
    raise ValueError("Match format must be Bo1, Bo3, or Bo5")

def predict_match(team_a, team_b, best_of):
    scores = score_probabilities(game_probability(team_a, team_b), best_of)
    return sum(prob for (wins_a, wins_b), prob in scores.items() if wins_a > wins_b)

st.markdown("""

<style>

.stApp {background-color: #0B1020; color: #E8EDF5;}

.block-container {max-width: 1100px; padding-top: 2.5rem; padding-bottom: 4rem;}

h1, h2, h3 {color: #F1F5F9 !important;}

[data-testid="stMetric"] {background-color: #151D30; border: 1px solid #28354A;

    border-radius: 14px; padding: 20px; text-align: center;}

[data-testid="stMetricValue"] {color: #D8B76A; font-weight: 700;}

[data-testid="stMetricLabel"] {color: #B6C3D6;}

hr {border-color: #28354A;}

.subtitle {color: #9CAFC6; font-size: 1.05rem; margin-top: -0.6rem; margin-bottom: 2rem;}

.section-label {color: #D8B76A; font-size: .8rem; font-weight: 700;

    letter-spacing: .15em; text-transform: uppercase; margin-bottom: .8rem;}

.team-acronym {text-align: center; font-size: 1.5rem; font-weight: 700; color: #F1F5F9;}

.team-region {text-align: center; color: #9CAFC6; font-size: .85rem;}

.score-label {text-align: center; color: #9CAFC6; font-size: .75rem;

    font-weight: 700; letter-spacing: .1em;}

.score-value {text-align: center; color: #F1F5F9; font-size: 2.5rem; font-weight: 800;}

.score-prob {text-align: center; color: #D8B76A; font-size: .85rem;}

.footer {text-align: center; color: #7889A3; font-size: .8rem; margin-top: 3rem;}

</style>

""", unsafe_allow_html=True)

st.title("LoL Worlds 2026 Predictor")
st.markdown('<p class="subtitle">Elo-powered match predictions for the League of Legends 2026 World Championship.</p>', unsafe_allow_html=True)

if missing_teams:
    st.warning("Missing Elo ratings for: " + ", ".join(missing_teams) + ". Check the team aliases in app.py.")
if len(teams) < 2:
    st.error("At least two Worlds teams need Elo ratings.")
    st.stop()

st.divider()

st.markdown('<p class="section-label">Worlds 2026 Match Predictor</p>', unsafe_allow_html=True)
col_a, col_vs, col_b = st.columns([5, 1, 5], vertical_alignment="bottom")
with col_a:
    team_a = st.selectbox("Team A", teams, index=teams.index("T1") if "T1" in teams else 0, format_func=team_label)
with col_vs:
    st.markdown('<div style="text-align:center;color:#D8B76A;font-weight:800;padding-bottom:12px">VS</div>', unsafe_allow_html=True)
with col_b:
    team_b = st.selectbox("Team B", teams, index=teams.index("Gen.G") if "Gen.G" in teams else min(1, len(teams)-1), format_func=team_label)
format_label = st.segmented_control("Match Format", options=["Bo1", "Bo3", "Bo5"], default="Bo5")
best_of = {"Bo1": 1, "Bo3": 3, "Bo5": 5}[format_label or "Bo5"]

st.divider()

st.markdown('<p class="section-label">Predicted Match Outcome</p>', unsafe_allow_html=True)
if team_a == team_b:
    st.warning("Please select two different teams.")
else:
    p_game = game_probability(team_a, team_b)
    scores = score_probabilities(p_game, best_of)
    p_a = predict_match(team_a, team_b, best_of)
    p_b = 1 - p_a
    most_likely_score, exact_prob = max(scores.items(), key=lambda item: item[1])
    left, middle, right = st.columns([4, 3, 4], vertical_alignment="center")
    with left:
        logo_a = get_team_logo(team_a)
        if logo_a:
            _, centered, _ = st.columns([1, 2, 1])
            with centered:
                st.image(str(logo_a), use_container_width=True)
        st.markdown(f'<div class="team-acronym">{acronym(team_a)}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="team-region">{team_regions[team_a]}</div>', unsafe_allow_html=True)
        st.metric("Series Win Probability", f"{p_a:.1%}")
        st.caption(f"Elo: {worlds_ratings[team_a]:,.0f}")
    with middle:
        st.markdown('<div class="score-label">MOST LIKELY SCORE</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="score-value">{most_likely_score[0]} – {most_likely_score[1]}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="score-prob">{exact_prob:.1%} chance of this exact score</div>', unsafe_allow_html=True)
    with right:
        logo_b = get_team_logo(team_b)
        if logo_b:
            _, centered, _ = st.columns([1, 2, 1])
            with centered:
                st.image(str(logo_b), use_container_width=True)
        st.markdown(f'<div class="team-acronym">{acronym(team_b)}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="team-region">{team_regions[team_b]}</div>', unsafe_allow_html=True)
        st.metric("Series Win Probability", f"{p_b:.1%}")
        st.caption(f"Elo: {worlds_ratings[team_b]:,.0f}")
    st.progress(p_a)
    if abs(p_a - .5) < 1e-10:
        st.info("Both teams are evenly matched.")
    else:
        favourite = team_a if p_a > .5 else team_b
        st.markdown(f"**{acronym(favourite)}** is favoured to win this {format_label}.")
    with st.expander("All Possible Scores"):
        st.caption("Each percentage is the probability of that exact final score, not the overall series win probability.")
        ordered_scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        for (wins_a, wins_b), probability in ordered_scores:
            label_col, bar_col, value_col = st.columns([2, 5, 1], vertical_alignment="center")
            with label_col:
                st.write(f"{acronym(team_a)} {wins_a}–{wins_b} {acronym(team_b)}")
            with bar_col:
                st.progress(probability)
            with value_col:
                st.write(f"{probability:.1%}")
    st.caption("Series and exact-score probabilities assume independent games with a constant game-level win probability; they do not model side changes, drafts, or within-series Elo updates.")

st.divider()

# ==========================================
# WORLDS KNOCKOUT BRACKET SIMULATOR
# ==========================================

import html
import streamlit.components.v1 as components

st.markdown('<p class="section-label">Worlds 2026 Knockout Bracket</p>', unsafe_allow_html=True)
st.caption("Custom knockout simulation. Choose eight teams and select winners below; the tree updates automatically. This is not the official knockout draw.")

seeded = sorted(teams, key=lambda t: worlds_ratings[t], reverse=True)[:8]
with st.expander("Choose the eight quarterfinalists"):
    slots = st.columns(4)
    bracket_teams = []
    for i in range(8):
        with slots[i % 4]:
            bracket_teams.append(st.selectbox(
                f"Seed {i + 1}", teams, index=teams.index(seeded[i]),
                format_func=acronym, key=f"seed_{i}"
            ))

def choose_winner(first, second, key, label):
    probability = predict_match(first, second, 5)
    favourite = first if probability >= 0.5 else second
    return st.selectbox(
        label, [first, second], index=0 if favourite == first else 1,
        format_func=acronym, key=f"pick_{key}_{first}_{second}"
    )

if len(set(bracket_teams)) != 8:
    st.warning("Choose eight different quarterfinalists.")
else:
    st.markdown("**Choose who advances**")
    qf_cols = st.columns(4)
    qf_winners = []
    for i in range(4):
        with qf_cols[i]:
            qf_winners.append(choose_winner(
                bracket_teams[2*i], bracket_teams[2*i+1], f"qf{i}", f"Quarterfinal {i+1}"
            ))

    sf_cols = st.columns(2)
    sf_winners = []
    for i in range(2):
        with sf_cols[i]:
            sf_winners.append(choose_winner(
                qf_winners[2*i], qf_winners[2*i+1], f"sf{i}", f"Semifinal {i+1}"
            ))
    champion = choose_winner(sf_winners[0], sf_winners[1], "final", "Grand Final")

    # Compact, responsive knockout tree. Logos are embedded in SVG so the
    # graphic also works on Streamlit Community Cloud without external URLs.
    def logo_svg(team, x, y, size=30):
        url = get_logo_data_url(team)
        if not url:
            return ""
        return (f'<image href="{url}" x="{x}" y="{y}" width="{size}" '
                f'height="{size}" preserveAspectRatio="xMidYMid meet"/>')

    def svg_match(x, y, first, second, winner, title, width=205):
        prob = predict_match(first, second, 5)
        rows = [(first, prob), (second, 1-prob)]
        out = [f'<rect x="{x}" y="{y}" width="{width}" height="100" rx="10" '
               'fill="#151D30" stroke="#36445A" stroke-width="1.2"/>',
               f'<text x="{x+11}" y="{y+16}" class="match-title">{html.escape(title)}</text>']
        for j, (team, chance) in enumerate(rows):
            top = y + 24 + j * 36
            selected = team == winner
            color = "#EAC776" if selected else "#E8EDF5"
            if selected:
                out.append(f'<rect x="{x+5}" y="{top-2}" width="{width-10}" height="35" '
                           'rx="5" fill="#293042"/>')
            out.append(logo_svg(team, x+10, top+1, 29))
            out.append(f'<text x="{x+44}" y="{top+22}" fill="{color}" '
                       f'font-size="15" font-weight="{700 if selected else 500}">'
                       f'{html.escape(acronym(team))}</text>')
            out.append(f'<text x="{x+width-10}" y="{top+22}" text-anchor="end" '
                       f'fill="{color}" font-size="12" font-weight="600">{chance:.0%}</text>')
        return "".join(out)

    # Coordinates are arranged to use the full page width without scrolling.
    # Connector lines are drawn first, behind the match cards.
    qx, sx, fx, cx = 12, 272, 532, 792
    qf_y = [48, 202, 356, 510]
    sf_y = [125, 433]
    final_y = 279
    card_w = 205
    mid_q = qx + card_w + 24
    mid_s = sx + card_w + 24
    pieces = [
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1005 660" '
        'role="img" aria-label="Worlds knockout bracket with logos and advancing teams" '
        'style="display:block;width:100%;height:auto;font-family:\'Source Sans 3\',\'Source Sans Pro\',Arial,sans-serif;">',

        '<style>'
        'text{font-family:"Source Sans 3","Source Sans Pro",Arial,sans-serif;font-kerning:normal;}'
        '.round-title{fill:#D8B76A;font-size:15px;font-weight:700;letter-spacing:.7px;text-anchor:middle;}'
        '.match-title{fill:#9CAFC6;font-size:12px;font-weight:600;}'
        '</style>',

        '<rect width="1005" height="660" rx="12" fill="#0B1020"/>',

        f'<text x="{qx+card_w/2}" y="26" class="round-title">QUARTERFINALS</text>',
        f'<text x="{sx+card_w/2}" y="26" class="round-title">SEMIFINALS</text>',
        f'<text x="{fx+card_w/2}" y="26" class="round-title">FINAL</text>',
        f'<text x="{cx+199/2}" y="26" class="round-title">CHAMPION</text>',
    ]

    for i in range(4):
        source_y = qf_y[i] + 50
        target_y = sf_y[i // 2] + 50
        pieces.append(f'<path d="M{qx+card_w} {source_y} H{mid_q} V{target_y} H{sx}" '
                      'fill="none" stroke="#786746" stroke-width="2"/>')
    for i in range(2):
        source_y = sf_y[i] + 50
        target_y = final_y + 50
        pieces.append(f'<path d="M{sx+card_w} {source_y} H{mid_s} V{target_y} H{fx}" '
                      'fill="none" stroke="#786746" stroke-width="2"/>')
    pieces.append(f'<path d="M{fx+card_w} {final_y+50} H{cx}" '
                  'stroke="#D8B76A" stroke-width="2"/>')
    for i in range(4):
        pieces.append(svg_match(qx, qf_y[i], bracket_teams[2*i],
                                bracket_teams[2*i+1], qf_winners[i], f"QF {i+1}"))
    for i in range(2):
        pieces.append(svg_match(sx, sf_y[i], qf_winners[2*i],
                                qf_winners[2*i+1], sf_winners[i], f"SF {i+1}"))
    pieces.append(svg_match(fx, final_y, sf_winners[0], sf_winners[1], champion, "GRAND FINAL"))
    pieces.append(f'<rect x="{cx}" y="{final_y-3}" width="199" height="108" '
                  'rx="12" fill="#29261E" stroke="#D8B76A" stroke-width="2"/>')
    pieces.append(logo_svg(champion, cx+12, final_y+21, 54))
    pieces.append(f'<text x="{cx+76}" y="{final_y+51}" fill="#D8B76A" '
                f'font-size="24" font-weight="800">{html.escape(acronym(champion))}</text>')
    pieces.append(f'<text x="{cx+12}" y="{final_y+85}" fill="#E8EDF5" '
                'font-size="12" font-weight="700">WORLD CHAMPION</text>')
    pieces.append('</svg>')
    svg = "".join(pieces)
    components.html(
        ('<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Source+Sans+3:wght@400;500;600;700&display=swap">'
         f'<div style="width:100%;margin:0;padding:0;overflow:hidden">{svg}</div>'),
        height=660,
        scrolling=False,
    )
    st.caption("The displayed probabilities are Bo5 estimates. Selecting winners overrides Elo favourites; the champion is your simulated outcome, not a title probability.")

st.divider()

st.markdown('<p class="section-label">Worlds 2026 Elo Leaderboard</p>', unsafe_allow_html=True)
leaderboard = pd.DataFrame([
    {"Logo": get_logo_data_url(team), "Team": acronym(team), "Region": team_regions[team], "Elo Rating": rating}
    for team, rating in worlds_ratings.items()
]).sort_values("Elo Rating", ascending=False).reset_index(drop=True)
leaderboard.insert(0, "Rank", range(1, len(leaderboard) + 1))
search = st.text_input("Search teams", placeholder="Search by acronym or region...")
if search:
    leaderboard = leaderboard[
        leaderboard["Team"].str.contains(search, case=False, na=False, regex=False)
        | leaderboard["Region"].str.contains(search, case=False, na=False, regex=False)
    ]
st.dataframe(
    leaderboard, hide_index=True, use_container_width=True, height=550, row_height=55,
    column_config={
        "Rank": st.column_config.NumberColumn("Rank", format="%d", width="small"),
        "Logo": st.column_config.ImageColumn("Logo", width="small"),
        "Team": st.column_config.TextColumn("Team", width="medium"),
        "Region": st.column_config.TextColumn("Region", width="small"),
        "Elo Rating": st.column_config.NumberColumn("Elo Rating", format="%.1f", width="medium"),
    },
)

st.divider()

with st.expander("About this Project"):
    st.markdown(r"""

This is a small side project I decided to build for two reasons.

First, I play a lot of League of Legends, and with Worlds 2026 coming around, I wanted to see whether I could improve my Pick'ems predictions using statistics rather than relying entirely on my gut feeling.

Second, I stumbled across Oracle's Elixir while brainstorming ideas for my CS50P final project. I never ended up using it back then, but I've always wanted to do something with the data.

So I figured, why not combine the two?

This project uses a simple Elo rating system to estimate team strength from historical match results. It's definitely not comprehensive since League matches are influenced by so many factors that Elo cannot capture, but I thought it would still be fun to see how far a simple statistical model like this could go in predicting match outcomes.

### Elo Rating System

This predictor uses a baseline Elo rating system trained on professional League of Legends match data from Oracle's Elixir (2024–2026).

The model incorporates:

\- Game-level Elo updates

\- Stage-dependent K-factor multipliers

\- Offseason rating regression

\- Historical regional and international match results

The expected single-game win probability is:

$$

P(A) = \frac{1}{1 + 10^{(R_B-R_A)/400}}

$$

### Model Limitations

This is a simple baseline model rather than a comprehensive prediction system. It does not directly account for champion drafts, patch changes, individual player performance, roster changes, or team strategies.

Bo3 and Bo5 predictions assume independent games with constant game-level win probabilities.

**Predictions are statistical estimates, not guaranteed outcomes.**

**Data source:** Oracle's Elixir

""")
st.markdown('<p class="footer">LoL Worlds 2026 Predictor | Built with Python and Streamlit</p>', unsafe_allow_html=True)
