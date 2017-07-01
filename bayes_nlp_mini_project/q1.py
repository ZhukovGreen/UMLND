s = 'so if you could just go ahead and pack up your stuff" and move it down there that would be terrific OK'
words = [str(_s).lower() for _s in s.split(' ')]
print(words.count('if'))
print(words.__len__())