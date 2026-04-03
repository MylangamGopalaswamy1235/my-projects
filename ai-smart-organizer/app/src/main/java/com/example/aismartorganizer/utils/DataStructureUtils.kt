package com.example.aismartorganizer.utils

import com.example.aismartorganizer.data.NoteEntity
import com.example.aismartorganizer.data.TaskEntity
import java.util.LinkedList
import java.util.Queue

object DataStructureUtils {

    // Graph: simulate related notes by same first word in title.
    fun buildRelatedGraph(notes: List<NoteEntity>): Map<Int, List<Int>> {
        val graph = HashMap<Int, MutableList<Int>>()
        notes.forEach { source ->
            val srcKey = source.title.split(" ").firstOrNull()?.lowercase() ?: ""
            notes.filter { it.id != source.id }.forEach { target ->
                val targetKey = target.title.split(" ").firstOrNull()?.lowercase() ?: ""
                if (srcKey.isNotBlank() && srcKey == targetKey) {
                    graph.getOrPut(source.id) { mutableListOf() }.add(target.id)
                }
            }
        }
        return graph
    }

    // Tree: group notes by parentId.
    fun flattenTree(notes: List<NoteEntity>): List<Pair<NoteEntity, Int>> {
        val childrenMap = notes.groupBy { it.parentId }
        val result = mutableListOf<Pair<NoteEntity, Int>>()

        fun dfs(parentId: Int?, depth: Int) {
            childrenMap[parentId].orEmpty().forEach {
                result.add(it to depth)
                dfs(it.id, depth + 1)
            }
        }

        dfs(null, 0)
        return result
    }

    // Queue: sort by priority and expose queue order text.
    fun prioritizeTasks(tasks: List<TaskEntity>): Queue<TaskEntity> {
        val sorted = tasks.sortedBy {
            when (it.priority) {
                "High" -> 0
                "Medium" -> 1
                else -> 2
            }
        }
        return LinkedList(sorted)
    }
}
