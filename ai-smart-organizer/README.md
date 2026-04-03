# AI Smart Knowledge Organizer with Task Intelligence and Insights

Beginner-friendly Android app (Kotlin + XML, MVVM, Fragments, Room) for Android 13+.

## Modules
1. Notes
2. Knowledge Map
3. To-Do
4. AI Assistant
5. Insights

## Data Structures used
- **Tree**: Hierarchy of notes in Knowledge Map
- **Graph**: Related notes mapping
- **Queue**: Smart task priority ordering
- **Stack**: Basic undo flow for note creation state
- **HashMap**: Frequency analysis and insight counters

## Project Structure
```
ai-smart-organizer/
├── app/
│   ├── src/main/java/com/example/aismartorganizer/
│   │   ├── data/ (Room entities, DAOs, database, repository)
│   │   ├── adapter/ (RecyclerView adapters)
│   │   ├── ui/
│   │   │   ├── notes/
│   │   │   ├── knowledgemap/
│   │   │   ├── todo/
│   │   │   ├── assistant/
│   │   │   └── insights/
│   │   ├── utils/ (DS + insights helpers)
│   │   ├── viewmodel/
│   │   ├── MainActivity.kt
│   │   └── SmartOrganizerApp.kt
│   └── src/main/res/layout/ (all XML screens + item rows)
└── build files
```

## Build Config
- Min SDK: 33
- Target/Compile SDK: 35
- Room DB entities: `NoteEntity`, `TaskEntity`

## Notes
- AI tools are simulated/static (no external API).
- Attachment actions are UI placeholders.
