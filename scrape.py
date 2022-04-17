from Scweet.scweet import scrape
from Scweet.user import get_user_information, get_users_following, get_users_followers

data = scrape(words=None, since="2022-01-01", until="2022-04-30", from_account ="etribune" ,         interval=1, headless=False, display_type="Top", save_images=False, lang="en",
	resume=False, filter_replies=False, proximity=False )