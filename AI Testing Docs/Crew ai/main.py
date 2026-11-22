# Agents , Tasks and Tools (LLM) - llama3, Crew ai

from pathlib import Path

from crewai import Agent, Task, LLM, Process, Crew
from youtube_transcript_api import YouTubeTranscriptApi

#All about youtube transcript
video_id = "PFvGRxZb2dU" 
saved_copy = Path("transcript.txt")
if saved_copy.exists():
    transcript = saved_copy.read_text(encoding="utf-8-sig")
else:
    captions = YouTubeTranscriptApi().fetch(video_id, languages=["en","hi"])
    transcript = "".join(line.text for line in captions)
if not transcript.strip():
    raise SystemExit("Transcript is empty") 

#Agents
llm = LLM(model="ollama/llama3:latest", base_url="http://localhost:11434")
researcher = Agent(llm=llm, role="Video Researcher", 
                   goal="Identify the main ideas and examples in the transcript.", 
                   backstory="You take accurate notes on what the speaker says.")
writer = Agent(llm=llm, role="Content Writer", 
                   goal="Explain the research notes in plain English.", 
                   backstory="You are skilled at synthesizing information and creating clear, concise summaries.")

#tasks
research = Task(agent=researcher, 
                    description="Take notes using only this transcript:\n{transcript}.", 
                    expected_output="A list of main ideas and examples from the transcript.") 

write = Task(agent=writer, 
                    description="Use the research notes to summarize the video in English.", 
                    expected_output="A markdown summary with one overview paragragh and key takeaways",
                    context=[research],
                    output_file="summary.md") 

crew = Crew(agents=[researcher, writer],tasks=[research, write] ,process=Process.sequential)
result = crew.kickoff(inputs={"transcript": transcript})
print(result.raw)
