import pandas as pd
import language_tool_python

tool = language_tool_python.LanguageTool('en-US')
data = pd.read_csv('./train.csv')
def grammercheck(df):
    counts = {'counts':[]}
    for i in range(2):
        matches = tool.check(df['full_text'][i])
        counts['counts'].append(matches)
    return counts

counts = grammercheck(data)
print(counts)
#matches = tool.check(text)