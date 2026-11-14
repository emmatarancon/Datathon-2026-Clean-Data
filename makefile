# Uso: make canvi "tu mensaje aqui"
canvi:
	@if [ -z "$(filter-out $@,$(MAKECMDGOALS))" ]; then \
		echo "Error: falta el missatge. Usa: make canvi 'missatge a enviar'"; \
		exit 1; \
	fi
	git add .
	@if git diff --cached --quiet; then \
		echo "No hi ha canvis nous per guardar (working tree clean)."; \
	else \
		git commit -m "$(filter-out $@,$(MAKECMDGOALS))"; \
	fi

puja:
	git pull --rebase
	git push

baixa:
	git pull --rebase

%:
	@: