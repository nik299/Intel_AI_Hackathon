global state
global dep_score
global feeling# can be done classifing anxiety and depressing statements 
def quitionare(query,rating):
  global state
  global dep_score
  global feeling
  if query == 'quit':
    state=0
  if state == 0:
    state =1
    return "Hello I am Adam.How are you felling now"
  elif state == 1:
    if feeling == 0:#good
      state=0
      return "have a nice day"
    elif feeling ==1:#depressed
      state = 2
      return "I want to ask you more about this reply with yes(no) if you want(don't want) to anwer,over the last 2 weeks, how often have you been bothered by any of the following problems?"
    elif feeling ==2:#anxious
      state=3
      return "ok I am going to ask some questions"
    else:
      return 'sorry didnot got it'
  elif state == 2:
    if query == 'yes':
      state = 21
      dep_score=0
      return 'having Little interest or pleasure in doing things.'
    if query =='no':
      state = 0
      return 'ok bye'
  elif state == 21:
    dep_score+=rating
    state = 22
    return 'ok next one:Are you feeling down, depressed, or hopeless'
  elif state == 22:
    dep_score+=rating
    state+=1
    return 'ok next one:Are you having Trouble falling or staying asleep, or sleeping too much'
  elif state == 23:
    dep_score+=rating
    state+=1
    return 'ok next one:Are you feeling tired or having little energy'
  elif state == 24:
    dep_score+=rating
    state+=1
    return 'ok next one:did you have poor appetite or overeating these 2 weeks'
  elif state == 25:
    dep_score+=rating
    state+=1
    return 'ok next one:did you feel bad about yourself - or that you are a failure or have let yourself or your family down'
  elif state == 26:
    dep_score+=rating
    state+=1
    return 'ok next one:did you had any trouble concentrating on things, such as reading the newspaper or watching television'
  elif state == 27:
    dep_score+=rating
    state+=1
    return 'ok next one:did you move or speak so slowly that other people could have noticed'
  elif state == 28:
    dep_score+=rating
    state+=1
    return 'ok last one:did you had any thoughts that you would be better off dead, or of hurting yourself'
  elif state == 29:
    state=0
    dep_score+=rating
    if dep_score == 0:
      return 'you are  having no depression problem'
    elif dep_score < 5:
      return 'be positve It is not a big issue'
    elif dep_score < 10:
      return "Don't be so sad.You are having mild depression"  
    elif dep_score < 15:
      return 'stay strong you can face take help of your friends for suppot'
    elif dep_score < 20:
      return 'ok take car and its better to meet a professional' 
    elif dep_score > 19:
      return 'Please meet a doctor immediatly'
  elif state == 3:
    if query == 'yes':
      state = 31
      dep_score=0
      return 'Are you feeling nervous, anxious or on edge these 2 weeks'
    if query =='no':
      state = 0
      return 'ok bye'
  elif state == 31:
    dep_score+=rating
    state = 32
    return 'ok next one:Are you unable to stop or control worrying'
  elif state == 32:
    dep_score+=rating
    state+=1
    return 'ok next one:Are you worrying too much about different things'
  elif state == 33:
    dep_score+=rating
    state+=1
    return 'ok next one:Are you having trouble relaxing'
  elif state == 34:
    dep_score+=rating
    state+=1
    return 'ok next one:Are you being so restless that it is hard to sit still'
  elif state == 35:
    dep_score+=rating
    state+=1
    return 'ok next one:Are you being easily annoyed or irritable'
  elif state == 36:
    dep_score+=rating
    state+=1
    return 'ok last one:are you feeling afraid, as if something awful might happen'
  elif state == 37:
    state=0
    dep_score+=rating
    if dep_score == 0:
      return 'you are  having no anxiety problem'
    elif dep_score < 5:
      return 'be cool It is not a big issue'
    elif dep_score < 10:
      return "Don't be so sad.You are having mild anxiety"  
    elif dep_score < 15:
      return 'stay strong you can face take help of you may consult a doctor'
    elif dep_score > 14:
      return 'ok take care and its better to meet a professional' 


state =0

depressed=['depressed','low','sad','sucide','die','kil','jump','hang']
anxious =['anxious','tensed','worry','worrying']

if state=0:
  for d in query.split():
    if d in depressed:
      feeling=1
      break
    if d in anxious:
      feeling=2
      break
    else:
      feeling=0
# rating= int(squery.split()[1])
# query=squery[4:]
# quitionare(query,rating)

