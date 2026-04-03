package com.example.aismartorganizer.ui.assistant

import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import androidx.fragment.app.Fragment
import com.example.aismartorganizer.databinding.FragmentAiAssistantBinding

class AIAssistantFragment : Fragment() {

    private var _binding: FragmentAiAssistantBinding? = null
    private val binding get() = _binding!!

    override fun onCreateView(inflater: LayoutInflater, container: ViewGroup?, savedInstanceState: Bundle?): View {
        _binding = FragmentAiAssistantBinding.inflate(inflater, container, false)
        return binding.root
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        binding.btnAsk.setOnClickListener {
            val query = binding.inputQuery.text.toString().trim()
            binding.tvResponse.text = when {
                query.contains("dbms", true) -> "Tip: Revise normalization and indexing first."
                query.contains("pending", true) -> "You have pending tasks. Start from High priority queue item."
                query.isBlank() -> "Try asking: 'Complete pending tasks'"
                else -> "Simulated AI: break your goal into 3 small actions and set deadlines."
            }
        }
    }

    override fun onDestroyView() {
        super.onDestroyView()
        _binding = null
    }
}
