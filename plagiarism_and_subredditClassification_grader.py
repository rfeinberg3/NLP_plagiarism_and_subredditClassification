import random
import traceback
import time
import gensim.models.keyedvectors as word2vec
import json


studentName = "TestStudent"
inputFileName = 'plagiarism_and_subredditClassification.py'

w2v_vectorFile = "GoogleNews-vectors-negative300.bin"

redditCommentsTrainingFile = "redditComments_train.jsonlist"
redditCommentsTestFile = "redditComments_test_notGraded.jsonlist"

#problems for task 1
problems = [\
	(
	  [
	    "On the second voyage, with his ship beset by storms south of Cape Horn, Hatley shot an albatross, an incident immortalised by Samuel Taylor Coleridge in his poem, The Rime of the Ancient Mariner.",
	    " Born in 1685 in Woodstock, Oxfordshire, Hatley went to sea in 1708 as part of Woodes Rogers's expedition against the Spanish.",
	    " Rogers circumnavigated the world, but Hatley was captured on the coast of present-day Ecuador and imprisoned in Lima, where he was tortured by the Inquisition.",
	    " He was released and returned to Britain in 1713.",
	    " Hatley's second voyage, under George Shelvocke, was the source of the albatross incident, and also ended with his capture by the Spanish.",
	    " As Hatley had, at Shelvocke's direction, looted a Portuguese vessel on the coast of Brazil, the Spanish this time held him as a pirate, though ultimately they released him again, deciding that Shelvocke was the more culpable party.",
	    " Hatley returned to Britain in 1723, and hastily sailed to Jamaica to avoid trial for piracy.",
	    " His fate thereafter is unknown.",
	  ],
	  "On the subsequent voyage, with his ship assailed by tempests south of Cape Horn, Hatley shot an albatross, an occurrence deified by Samuel Taylor Coleridge in his poem, The Rime of the Ancient Mariner."
	),
	(
	  [
	    "Much of what is known about Hatley's subsequent life is in connection with the two privateering voyages that he made to the Pacific coast of South America.",
	    " Privateers were men who sailed in armed merchant ships carrying letters of marque from their government authorising them to plunder foreign enemies, keeping any profits for themselves and their ships' owners.",
	    " The first such voyage made by Hatley was under the command of Captain Woodes Rogers during the War of the Spanish Succession, which found Britain and Spain on opposing sides.",
	    " In 1708, at the age of twenty-three, he signed on as third mate (a junior officer position) of the Duchess, the smaller of Rogers's two ships, the other being the Duke.",
	    " Rogers's vessels were then being readied in Bristol for a long and difficult journey to the Pacific coast of South America.",
	    " The purpose of the Rogers expedition was to go around Cape Horn into the South Pacific, to damage Spanish settlements and interests along the South American Pacific coast, and to capture booty for their own profit, including the large treasure galleons that sailed from Manila to Mexico.",
	    " The two ships were to be crammed with men, supplies to maintain them, and with guns and powder, for the success of the expedition depended on being able to outfight those vessels they sought to capture and plunder.",
	  ],
	  "Privateers were men who cruised in outfitted dealer boats conveying letters of marque from their administration approving them to loot outside foes, keeping any benefits for themselves and their boats' proprietors."
	),
	(
	  [
	    "The process begins as two continents (different bits of continental crust), separated across a tract of ocean (and oceanic crust), approach each other, while the oceanic crust is slowly consumed at a subduction zone.",
	    " The subduction zone runs along the edge of one of the continents and dips under it, raising volcanic mountain chains at some distance behind it, such as the Andes of South America today.",
	    " Subduction involves the whole lithosphere, the density of which is largely controlled by the nature of the crust it carries.",
	    " Oceanic crust is thin (~6 km thick) and dense (about 3.",
	    "3 g/cm³), consisting of basalt, gabbro, and peridotite.",
	    " Consequently, most oceanic crust is subducted easily at an oceanic trench.",
	    " In contrast, continental crust is thick (~45 km thick) and buoyant, composed mostly of granitic rocks (average density about 2.",
	    "5 g/cm³).",
	    " Continental crust is subducted with difficulty, but is subducted to depths of 90-150 km or more, as evidenced by ultra-high pressure (UHP) metamorphic suites.",
	    " Normal subduction continues as long as the ocean exists, but the subduction system is disrupted as the continent carried by the downgoing plate enters the trench.",
	    " Because it contains thick continental crust, this lithosphere is less dense than the underlying asthenospheric mantle and normal subduction is disrupted.",
	    " The volcanic arc on the upper plate is slowly extinguished.",
	    " Resisting subduction, the crust buckles up and under, raising mountains where a trench used to be.",
	    " The position of the trench becomes a zone that marks the suture between the two continental terranes.",
	    " Suture zones are often marked by fragments of the pre-existing oceanic crust and mantle rocks, known as ophiolites.",
	  ],
	  "Subduction includes the entire lithosphere, the thickness of which is to a great extent constrained by the idea of the covering it conveys."
	),
	(
	  [
	    "Philological analysis of Archaic Latin works, such as those of Plautus, which contain snippets of everyday speech, indicates that a spoken language, Vulgar Latin (termed sermo vulgi, 'the speech of the masses', by Cicero), existed concurrently with literate Classical Latin.",
	    " The informal language was rarely written, so philologists have been left with only individual words and phrases cited by classical authors and those found as graffiti.",
	    " However, philologists have found traces of this seldom written language in the earlier works and drafts of William Shakespeare's many plays.",
	    " One of the examples most prominent is in one of Shakespeare's first drafts of Titus Andronicus, where Shakespeare referred to the villain of the play as a 'adipem pullum'.",
	    " (Fat chicken.",
	    ") As it was free to develop on its own, there is no reason to suppose that the speech was uniform either diachronically or geographically.",
	    " On the contrary, romanised European populations developed their own dialects of the language, which eventually led to the differentiation of Romance languages.",
	    " The decline of the Roman Empire meant a deterioration in educational standards that brought about Late Latin, a postclassical stage of the language seen in Christian writings of the time.",
	    " It was more in line with everyday speech, not only because of a decline in education but also because of a desire to spread the word to the masses.",
	  ],
	  "One of the models most unmistakable is in one of Shakespeare's first drafts of Titus Andronicus, where Shakespeare alluded to the antagonist of the play as an 'adipem pullum'."
	),
	(
	  [
	    "Medieval Latin is the written Latin in use during that portion of the postclassical period when no corresponding Latin vernacular existed.",
	    " The spoken language had developed into the various incipient Romance languages; however, in the educated and official world Latin continued without its natural spoken base.",
	    " Moreover, this Latin spread into lands that had never spoken Latin, such as the Germanic and Slavic nations.",
	    " It became useful for international communication between the member states of the Holy Roman Empire and its allies.",
	    " Without the institutions of the Roman empire that had supported its uniformity, medieval Latin lost its linguistic cohesion: for example, in classical Latin sum and eram are used as auxiliary verbs in the perfect and pluperfect passive, which are compound tenses.",
	    " Medieval Latin might use fui and fueram instead.",
	    " Furthermore, the meanings of many words have been changed and new vocabularies have been introduced from the vernacular.",
	    " Identifiable individual styles of classically incorrect Latin prevail.",
	  ],
	  "Medieval Latin is the composed Latin being used during that segment of the postclassical period when no relating Latin vernacular existed."
	),
	(
	  [
	    "An ongoing trans-national air pollution crisis is affecting several countries in Southeast Asia, including Brunei, Indonesia, Malaysia, Philippines, Singapore, Thailand, and Vietnam.",
	    " Thailand began to experience a haze in February that lasted until May, peaking in March and April.",
	    " Later in the year, starting from June to July, Indonesia began to experience haze.",
	    " Malaysia was affected from August, while Singapore, Brunei, and Vietnam experienced haze in September.",
	    " It is the latest occurrence of the Southeast Asian haze, a long-term issue that occurs in varying intensity during every dry season in the region.",
	    " It has mainly been caused by forest fires resulting from illegal slash-and-burn clearing performed on behalf of the palm oil industry in Indonesia, principally on the islands of Sumatra and Borneo, which then spread quickly in the dry season.",
	  ],
	  "Thailand started to encounter a cloudiness in February that went on until May, topping in March and April."
	),
	(
	  [
	    "The earthquake caused severe damage to 135 houses in Mirpur District, with a further 319 being partially damaged, most in Mirpur and just four in Bhimber District.",
	    " Two bridges were reported damaged and parts of several roads were affected, particularly 14 km of the Main Jatlan Road.",
	    " According to the chairman of Pakistan's National Disaster Management Authority (NDMA), 'In Mirpur, besides the city, a small town Jatlan, and two small villages Manda and Afzalpur' were among the worst-hit areas.",
	    " According to him, the main road which runs alongside a river from Mangla to Jatla suffered major damage.",
	    " According to the officials, the Mangla Dam, Pakistan's major water reservoir, was spared.",
	    " However, the dam's power house was closed, which resulted in a loss of 900 megawatts to Pakistan's national power grid.",
	    " At 7:20 pm, power generation at Mangla was resumed, restoring 700 MW to the national grid.",
	  ],
	  "As per the administrator of Pakistan's National Disaster Management Authority (NDMA), 'In Mirpur, other than the city, a community Jatlan, and two little towns Manda and Afzalpur' were among the most exceedingly awful hit regions."
	),
	(
	  [
	    "The United Kingdom has a doctrine of parliamentary sovereignty, so the Supreme Court is much more limited in its powers of judicial review than the constitutional or supreme courts of some other countries.",
	    " It cannot overturn any primary legislation made by Parliament.",
	    " However, it can overturn secondary legislation if, for an example, that legislation is found to be ultra vires to the powers in primary legislation allowing it to be made.",
	    " Further, under section 4 of the Human Rights Act 1998, the Supreme Court, like some other courts in the United Kingdom, may make a declaration of incompatibility, indicating that it believes that the legislation subject to the declaration is incompatible with one of the rights in the European Convention on Human Rights.",
	    " Such a declaration can apply to primary or secondary legislation.",
	    " The legislation is not overturned by the declaration, and neither Parliament nor the government is required to agree with any such declaration.",
	    " However, if they do accept a declaration, ministers can exercise powers under section 10 of the Human Rights Act to amend the legislation by statutory instrument to remove the incompatibility or ask Parliament to amend the legislation.",
	  ],
	  "Further, under segment 4 of the Human Rights Act 1998, the Supreme Court, similar to some different courts in the United Kingdom, may make an announcement of contrariness, demonstrating that it accepts that the enactment subject to the statement is contrary with one of the rights in the European Convention on Human Rights."
	),
	(
	  [
	    "The first case heard by the Supreme Court was HM Treasury v Ahmed, which concerned 'the separation of powers', according to Phillips, its inaugural President.",
	    " At issue was the extent to which Parliament has, by the United Nations Act 1946, delegated to the executive the power to legislate.",
	    " Resolution of this issue depended upon the approach properly to be adopted by the court in interpreting legislation which may affect fundamental rights at common law or under the European Convention on Human Rights.",
	    " Because of the doctrine of parliamentary sovereignty, the Supreme Court is much more limited in its powers of judicial review than the constitutional or supreme courts of some other countries.",
	    " It cannot overturn any primary legislation made by Parliament.",
	    " However, it can overturn secondary legislation if, for example, that legislation is found to be ultra vires to the powers in primary legislation allowing it to be made.",
	    " Further, under section 4 of the Human Rights Act 1998, the Supreme Court, like some other courts in the United Kingdom, may make a declaration of incompatibility, indicating that it believes that the legislation subject to the declaration is incompatible with one of the rights in the European Convention on Human Rights.",
	    " Such a declaration can apply to primary or secondary legislation.",
	    " The legislation is not overturned by the declaration, and neither Parliament nor the government is required to agree with any such declaration.",
	    " However, if they do accept a declaration, ministers can exercise powers under section 10 of the act to amend the legislation by statutory instrument to remove the incompatibility or ask Parliament to amend the legislation.",
	  ],
	  "The principal case heard by the Supreme Court was HM Treasury v Ahmed, which concerned 'the partition of forces', as indicated by Phillips, its debut President."
	),
	(
	  [
	    "The court is housed in Middlesex Guildhall - which it shares with the Judicial Committee of the Privy Council—in the City of Westminster.",
	    " The Constitutional Reform Act 2005 gave time for a suitable building to be found and fitted out before the Law Lords moved out of the Houses of Parliament, where they had previously used a series of rooms in the Palace of Westminster.",
	    " After a lengthy survey of suitable sites, including Somerset House, the Government announced that the new court would be at the Middlesex Guildhall, in Parliament Square, Westminster.",
	    " That decision was examined by the Constitutional Affairs Committee, and the grant of planning permission by Westminster City Council for refurbishment works was challenged in a judicial review by the conservation group Save Britain's Heritage.",
	    " It was also reported that English Heritage had been put under great pressure to approve the alterations.",
	    " Feilden + Mawson, supported by Foster & Partners, were the appointed architects.",
	  ],
	  "The Constitutional Reform Act 2005 gave time for an appropriate structure to be found and fitted out under the watchful eye of the Law Lords moved out of the Houses of Parliament, where they had recently utilized a progression of rooms in the Palace of Westminster."
	),
]
answers = [0,1,2,3,0,1,2,3,0,1]

#problems for task 2
with open(redditCommentsTestFile,'r', encoding='utf-8') as F:
	allComments = [json.loads(l) for l in F.readlines()]

outFile = open("grade_"+studentName+".txt", 'w', encoding='utf-8')

print("Loading w2v pretrained vectors...")
w2vModel = word2vec.KeyedVectors.load_word2vec_format\
	(w2v_vectorFile,binary=True)

def prnt(S):
	global outFile
	outFile.write(str(S) + "\n")
	print(S)

############################################################
############### TASK 1 #####################################
############################################################
"""This task is graded based on what fraction of the test set you get correct. If you get baselineCorrect, then you get 0. If you get maxCorrect, you get full credit. Extra credit is possible (see code for details).
"""
baselineCorrect = 0.7
maxCorrect = 0.9
fullCredit = 50
maxScore = 60

try:
	F = open(inputFileName, 'r', encoding='utf-8')
	exec("".join(F.readlines()))
except Exception as e:
	prnt("Couldn't open or execute '" + inputFileName + "': " + str(traceback.format_exc()))
	prnt("FINAL SCORE: 0")
	outFile.close()
	exit()

prnt("============================")
prnt("=========  TASK 1  =========")
prnt("============================")

try:
	prnt("CALLING YOUR setModel() FUNCTION")
	setModel(w2vModel)
except Exception as e:
	prnt("\tError arose: " + str(traceback.format_exc()))
	prnt("\tNOTE: We won't penalize you directly for this, but this is likely to lead to exceptions later.")

numCorrect = 0
for i in range(len(problems)):
	(sentences,target) = problems[i]
	A = answers[i]
	
	prnt("\n\nTESTING ON INPUT PROBLEM:")
	prnt("sentences:\n\t" + '\n\t'.join(sentences))
	prnt("target sentence: " + target)
	prnt("CORRECT OUTPUT:")
	prnt(str(A))
	prnt("YOUR OUTPUT:")
	try:
		startTime = time.time()
		result = findPlagiarism(sentences,target)
		prnt(result)
		endTime = time.time()		
		if endTime-startTime > 120:
			prnt("Time to execute was " + str(int(endTime-startTime)) + " seconds; this is too long (marked as wrong)")
		elif result==A:
			prnt("Correct!")
			numCorrect += 1
		else:
			prnt("Incorrect")

	except Exception as e:
		prnt("Marked as incorrect; there was an error while executing this problem: " + str(traceback.format_exc()))
percentCorrect = numCorrect*1.0/len(problems)
if maxCorrect == baselineCorrect: #avoid division by zero
	points = 0
else:
	points = min(maxScore, fullCredit * (percentCorrect - baselineCorrect) / (maxCorrect - baselineCorrect))
	points = max(0, points)
prnt("\nYou got " + str(percentCorrect*100) + "% correct: +" + str(points) + "/" + str(fullCredit) + " points")

totalScore = points
############################################################
############### TASK 2 #####################################
############################################################
prnt("============================")
prnt("=========  TASK 2  =========")
prnt("============================")

"""This task is graded based on what fraction of the test set you get correct. If you get baselineCorrect, then you get 0. If you get maxCorrect, you get full credit. Extra credit is possible (see code for details).
"""
baselineCorrect = 0.4
maxCorrect = 0.65
fullCredit = 50
maxScore = 60

penalty = 0
try:
	prnt("CALLING YOUR classifySubreddit_train() FUNCTION")
	startTime = time.time()
	classifySubreddit_train(redditCommentsTrainingFile)
	endTime = time.time()
except Exception as e:
	endTime = time.time()
	prnt("\tError arose: " + str(traceback.format_exc()))
	prnt("\tNOTE: We won't penalize you directly for this, but this is likely to lead to exceptions later.")
if endTime - startTime > 120:
	prnt("Time to execute was " + str(int(endTime-startTime)) + " seconds; this is too long (-10 points)")
	penalty += 10

numCorrect = 0
for comment in allComments:
	text = comment['body']
	sub = comment['subreddit']
	
	prnt("\n\nTESTING ON INPUT PROBLEM:")
	prnt("\t" + text)
	prnt("CORRECT OUTPUT:")
	print("\t" + sub)
	prnt("YOUR OUTPUT:")
	try:
		startTime = time.time()
		result = classifySubreddit_test(text)
		prnt("\t" + result)
		endTime = time.time()		
		if endTime-startTime > 120:
			prnt("Time to execute was " + str(int(endTime-startTime)) + " seconds; this is too long (marked as wrong)")
		elif result.lower()==sub.lower():
			prnt("Correct!")
			numCorrect += 1
		else:
			prnt("Incorrect")

	except Exception as e:
		prnt("Marked as incorrect; there was an error while executing this problem: " + str(traceback.format_exc()))
percentCorrect = numCorrect*1.0/len(allComments)
if maxCorrect == baselineCorrect: #avoid division by zero
	points = 0
else:
	points = min(maxScore, fullCredit * (percentCorrect - baselineCorrect) / (maxCorrect - baselineCorrect))
	points = max(0, points)
prnt("\nYou got " + str(percentCorrect*100) + "% correct: +" + str(points) + "/" + str(fullCredit) + " points")
if penalty != 0:
	points -= penalty
	prnt("After penalties: " + str(points))

totalScore += points
prnt("=============================")
prnt("=======  FINAL GRADE  =======")
prnt("=============================")
prnt(str(totalScore) + " / 100")

outFile.close()