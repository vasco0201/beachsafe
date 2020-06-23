import sys
import os
import argparse
import threading
import datetime
import numpy as np
import json
from pathlib import Path
from retrieve_images import *
from os import listdir
from safebeach_detector import *

# from add_data import *
WAIT_TIME_SECONDS = 5
flag = True



beachcams = [("Carcavelos", "https://video-auth1.iol.pt/beachcam/carcavelos/playlist.m3u8"),
			 ("Conceição Duquesa",
			  "https://video-auth1.iol.pt/beachcam/conceicao/playlist.m3u8"),
			 ("Baía de Cascais",
			  "https://video-auth1.iol.pt/beachcam/praiadospescadores/playlist.m3u8"),
			 ("S. João do Estoril",
			  "https://video-auth1.iol.pt/beachcam/saojoaodoestoril/playlist.m3u8"),
			 ("S. Pedro do Estoril",
			  "https://video-auth1.iol.pt/beachcam/saopedroestoria/playlist.m3u8"),
			 ("Parede", "https://video-auth1.iol.pt/beachcam/parede/playlist.m3u8")]

ticker = threading.Event()

if __name__ == '__main__':
		try:
			while not ticker.wait(WAIT_TIME_SECONDS):
				print("Retrieving images...")
				for i in range(len(beachcams)):
					ref = i
					curr = beachcams[i]
					f = open("data.txt", "a")
					print("Reference: ", ref, "\nBeach: ", curr[0])
					now = datetime.datetime.now().replace(microsecond=0).isoformat()
					# try:
					# 	os.mkdir(beachcam[0])
					# except:
					# 	pass
					print(curr[1])
					n_ppl, occupation = retrieve_process(
						beachcam=curr[1])
					print(n_ppl, occupation)
					data = {}
					# print(data)
					data = {
						"current": {
							"name": curr[0],
							"timestamp":now,
							},
						"people": {
							"mean": round(np.mean(n_ppl),2),
							"median": round(np.median(n_ppl),2),
							"max": round(np.max(n_ppl),2),
							"min": round(np.min(n_ppl),2)
						},
						"occupation":{
							"mean": round(np.mean(occupation),2),
							"median": round(np.median(occupation),2),
							"max": round(np.max(occupation),2),
							"min": round(np.min(occupation),2)
						}
					}
					print(data, file = f)

					f.close()
				print("Updating file...")
				if os.path.isdir('/content/drive'):
					copyfile('data.txt', '/content/drive/My Drive/data.txt')

				# 	if i == 1 and flag:
				# 		send_data(data, True)
				# 	else:
				# 		send_data(data, False)
				# flag = False
		except KeyboardInterrupt:
			pass
