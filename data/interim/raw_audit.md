# Step 01 — Raw CSV Data Audit

Rows inspected per file: up to 50,000

## `steam/0/games.csv`

- Kind: steam_games
- Sample rows: 4
- Sample covers whole file: True
- Columns: `id, name, numReviews, avgReviewScore, price, genres, tags, description, numFollowers`
- Empty cells (sample): `{}`
- unique_games_in_sample: 4
- duplicate_game_ids_in_sample: 0
- duplicate_casefold_titles_in_sample: 0

## `steam/20000/games.csv`

- Kind: steam_games
- Sample rows: 33,923
- Sample covers whole file: True
- Columns: `id, name, numReviews, avgReviewScore, price, genres, tags, description, numFollowers`
- Empty cells (sample): `{"description": 8}`
- unique_games_in_sample: 33,923
- duplicate_game_ids_in_sample: 0
- duplicate_casefold_titles_in_sample: 62

## `steam/39935/games.csv`

- Kind: steam_games
- Sample rows: 42
- Sample covers whole file: True
- Columns: `id, name, numReviews, avgReviewScore, price, genres, tags, description, numFollowers`
- Empty cells (sample): `{"description": 1}`
- unique_games_in_sample: 42
- duplicate_game_ids_in_sample: 0
- duplicate_casefold_titles_in_sample: 0

## `steam/59526/games.csv`

- Kind: steam_games
- Sample rows: 119
- Sample covers whole file: True
- Columns: `id, name, numReviews, avgReviewScore, price, genres, tags, description, numFollowers`
- Empty cells (sample): `{}`
- unique_games_in_sample: 119
- duplicate_game_ids_in_sample: 0
- duplicate_casefold_titles_in_sample: 0

## `steam/0/users_games.csv`

- Kind: steam_user_games
- Sample rows: 50,000
- Sample covers whole file: False
- Columns: `user_id, game_id, playtime_2weeks, playtime_forever`
- Empty cells (sample): `{}`
- unique_users_in_sample: 462
- unique_games_in_sample: 11,534
- duplicate_user_game_pairs_in_sample: 0
- playtime_2weeks (minutes): non_numeric_or_empty=0, zero_values=48,917, negative_values=0, median=0, p99=156.0, max=12,796
- playtime_forever (minutes): non_numeric_or_empty=0, zero_values=17,945, negative_values=0, median=40, p99=42046.3, max=776,153

## `steam/20000/users_games.csv`

- Kind: steam_user_games
- Sample rows: 50,000
- Sample covers whole file: False
- Columns: `user_id, game_id, playtime_2weeks, playtime_forever`
- Empty cells (sample): `{}`
- unique_users_in_sample: 33
- unique_games_in_sample: 10,819
- duplicate_user_game_pairs_in_sample: 1,077
- playtime_2weeks (minutes): non_numeric_or_empty=0, zero_values=49,930, negative_values=0, median=0, p99=0, max=18,601
- playtime_forever (minutes): non_numeric_or_empty=0, zero_values=22,866, negative_values=0, median=34, p99=2513.0, max=693,879

## `steam/39935/users_games.csv`

- Kind: steam_user_games
- Sample rows: 50,000
- Sample covers whole file: False
- Columns: `user_id, game_id, playtime_2weeks, playtime_forever`
- Empty cells (sample): `{}`
- unique_users_in_sample: 303
- unique_games_in_sample: 9,166
- duplicate_user_game_pairs_in_sample: 0
- playtime_2weeks (minutes): non_numeric_or_empty=0, zero_values=49,403, negative_values=0, median=0, p99=10, max=16,350
- playtime_forever (minutes): non_numeric_or_empty=0, zero_values=13,699, negative_values=0, median=128, p99=34981.7, max=936,486

## `steam/59526/users_games.csv`

- Kind: steam_user_games
- Sample rows: 50,000
- Sample covers whole file: False
- Columns: `user_id, game_id, playtime_2weeks, playtime_forever`
- Empty cells (sample): `{}`
- unique_users_in_sample: 816
- unique_games_in_sample: 5,683
- duplicate_user_game_pairs_in_sample: 0
- playtime_2weeks (minutes): non_numeric_or_empty=0, zero_values=48,157, negative_values=0, median=0, p99=491, max=19,992
- playtime_forever (minutes): non_numeric_or_empty=0, zero_values=14,501, negative_values=0, median=78, p99=73966.1, max=673,217

## `playstation/games.csv`

- Kind: playstation_games
- Sample rows: 23,151
- Sample covers whole file: True
- Columns: `gameid, title, platform, developers, publishers, genres, supported_languages, release_date`
- Empty cells (sample): `{"developers": 17, "publishers": 11, "genres": 142, "supported_languages": 12537}`
- unique_games_in_sample: 23,151
- duplicate_game_ids_in_sample: 0
- duplicate_casefold_titles_in_sample: 12,071

## `playstation/purchased_games.csv`

- Kind: playstation_purchases
- Sample rows: 46,582
- Sample covers whole file: True
- Columns: `playerid, library`
- Empty cells (sample): `{}`
- unique_users_in_sample: 46,582
- unique_games_in_sample: 29,296
- duplicate_user_ids_in_sample: 0
- empty_libraries_in_sample: 0
- unparsed_libraries_in_sample: 0
- duplicate_ids_inside_libraries_in_sample: 7
