package com.example.aismartorganizer.adapter

import android.view.LayoutInflater
import android.view.ViewGroup
import androidx.core.content.ContextCompat
import androidx.recyclerview.widget.RecyclerView
import com.example.aismartorganizer.data.NoteEntity
import com.example.aismartorganizer.databinding.ItemNoteBinding
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

class NotesAdapter(private val onClick: (NoteEntity) -> Unit) : RecyclerView.Adapter<NotesAdapter.NoteVH>() {
    private var items: List<NoteEntity> = emptyList()

    fun submitList(list: List<NoteEntity>) {
        items = list
        notifyDataSetChanged()
    }

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): NoteVH {
        val binding = ItemNoteBinding.inflate(LayoutInflater.from(parent.context), parent, false)
        return NoteVH(binding)
    }

    override fun onBindViewHolder(holder: NoteVH, position: Int) = holder.bind(items[position])
    override fun getItemCount(): Int = items.size

    inner class NoteVH(private val binding: ItemNoteBinding) : RecyclerView.ViewHolder(binding.root) {
        fun bind(note: NoteEntity) {
            binding.tvTitle.text = note.title
            binding.tvContent.text = note.content
            binding.tvTime.text = SimpleDateFormat("dd MMM yyyy, hh:mm a", Locale.getDefault()).format(Date(note.timestamp))
            binding.colorTag.setBackgroundColor(ContextCompat.getColor(binding.root.context, note.colorTag))
            binding.root.setOnClickListener { onClick(note) }
        }
    }
}
